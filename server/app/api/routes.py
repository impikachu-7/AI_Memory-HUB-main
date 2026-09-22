from fastapi.responses import RedirectResponse
from datetime import UTC, date, datetime, timedelta
import hmac
import json
import logging
import secrets
from urllib.parse import urlencode
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.core.security import decode_access_token, hash_password, verify_password
from app.database import get_db
from app.dependencies import AUTH_COOKIE_NAME, current_token, current_user
from app.models import AnalyticsEvent, Conversation, Memory, Message, ModelRegistry, OAuthIdentity, ProviderConfiguration, User, UserSettings
from app.repositories.owned import OwnedRepository
from app.schemas import *
from app.services.auth import clear_login_failures, consume_otp, create_session, issue_otp, login_allowed, record_login_failure, revoke_all_sessions, revoke_session
from app.services.credentials import decrypt_api_key, encrypt_api_key
from app.services.oauth import exchange_google_code, google_authorization_url
from app.services.memory_engine import extract_from_conversation, retrieve, vector_store
from app.services.llm import get_provider
from app.services.llm import context_builder
from app.services.llm.errors import ProviderError, provider_error
from app.services.data_export import export_conversations, export_memories, export_user_data

log = logging.getLogger(__name__)
OAUTH_STATE_COOKIE_NAME = "ai_memory_hub_oauth_state"


def secure_cookie_setting(settings) -> bool:
    if settings.app_env.lower() == 'production':
        return True
    local_origin = any(
        origin.strip().startswith(('http://localhost', 'http://127.0.0.1'))
        for origin in settings.frontend_origins.split(',')
    )
    return settings.cookie_secure and not local_origin


def cookie_samesite_setting(settings) -> str:
    return "none" if secure_cookie_setting(settings) else "lax"


def oauth_cookie_domain() -> str | None:
    # Authentication cookies are consumed by the backend, never by the Vercel
    # frontend. An incorrect COOKIE_DOMAIN must not make the browser discard them.
    return None


def clear_oauth_state_cookie(response: Response) -> None:
    for path in ("/", "/api/v1/auth/google"):
        response.delete_cookie(
            OAUTH_STATE_COOKIE_NAME,
            path=path,
            domain=oauth_cookie_domain(),
        )

router = APIRouter()
conversations, messages, memories, providers = (OwnedRepository(x) for x in (Conversation, Message, Memory, ProviderConfiguration))

def record_event(db: Session, user_id: str, event_type: str, provider: str | None = None, model_key: str | None = None, is_local: bool | None = None, metadata: dict | None = None):
    db.add(AnalyticsEvent(user_id=user_id, event_type=event_type, provider=provider, model_key=model_key, is_local=is_local, metadata_=metadata or {}))

def analytics_since(days: int | None) -> datetime | None:
    return datetime.now(UTC) - timedelta(days=days) if days else None

@router.post('/auth/register', status_code=202)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.scalar(select(User).where(User.email == body.email.lower()))
    if existing:
        if existing.is_email_verified: raise HTTPException(409, 'Email already registered')
        issue_otp(db, existing, 'verify_email')
        return {"detail": "Verification code sent"}
    user = User(email=body.email.lower(), password_hash=hash_password(body.password), full_name=body.full_name, is_active=False, is_email_verified=False)
    db.add(user); db.commit(); db.refresh(user); issue_otp(db, user, 'verify_email')
    return {"detail": "Verification code sent"}

@router.post('/auth/verify-email', response_model=TokenResponse)
def verify_email(body: OtpVerifyRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user: raise HTTPException(400, 'Invalid or expired code')
    consume_otp(db, user, 'verify_email', body.otp)
    user.is_email_verified, user.is_active = True, True; db.commit(); db.refresh(user)
    return TokenResponse(access_token=create_session(db, user), user=user)

@router.post('/auth/resend-verification', status_code=202)
def resend_verification(body: OtpRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user and not user.is_email_verified: issue_otp(db, user, 'verify_email')
    return {"detail": "If needed, a verification code has been sent"}

@router.post('/auth/login', response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    email = body.email.lower(); login_allowed(db, email)
    user = db.scalar(select(User).where(User.email == email))
    if not user or not user.is_email_verified or not user.is_active or not verify_password(body.password, user.password_hash):
        record_login_failure(db, email); raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Invalid email or password')
    clear_login_failures(db, email)
    return TokenResponse(access_token=create_session(db, user), user=user)

@router.post('/auth/logout', status_code=204)
def logout(response: Response, token: dict = Depends(current_token), db: Session = Depends(get_db)):
    revoke_session(db, token['sid'])
    response.delete_cookie(AUTH_COOKIE_NAME, path='/', domain=oauth_cookie_domain())

@router.post('/auth/forgot-password', status_code=202)
def forgot_password(body: OtpRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user and user.is_email_verified: issue_otp(db, user, 'reset_password')
    return {"detail": "If an account exists, a reset code has been sent"}

@router.post('/auth/verify-reset')
def verify_reset(body: OtpVerifyRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user: raise HTTPException(400, 'Invalid or expired code')
    consume_otp(db, user, 'reset_password', body.otp)
    from app.core.config import get_settings
    import jwt
    settings = get_settings()
    return {'reset_token': jwt.encode({'sub': user.id, 'purpose': 'password_reset', 'version': user.auth_version, 'exp': datetime.now(UTC) + timedelta(minutes=10)}, settings.jwt_secret, algorithm=settings.jwt_algorithm)}

@router.post('/auth/reset-password', status_code=204)
def reset_password(body: PasswordResetRequest, db: Session = Depends(get_db)):
    from app.core.config import get_settings
    import jwt
    try:
        claims = jwt.decode(body.reset_token, get_settings().jwt_secret, algorithms=[get_settings().jwt_algorithm])
        if claims.get('purpose') != 'password_reset': raise jwt.InvalidTokenError()
    except jwt.InvalidTokenError as exc: raise HTTPException(400, 'Invalid or expired reset token') from exc
    user = db.get(User, claims['sub'])
    if not user or claims.get('version') != user.auth_version: raise HTTPException(400, 'Invalid or expired reset token')
    user.password_hash = hash_password(body.new_password); user.auth_version += 1; db.commit(); revoke_all_sessions(db, user.id)

@router.get('/auth/me', response_model=UserRead)
def auth_me(user: User = Depends(current_user)): return user

@router.get('/auth/google/start')
def google_start():
    # Signed state is validated on callback; it contains no account credentials.
    from app.core.config import get_settings
    from jwt import encode
    settings = get_settings()
    state = encode({'nonce': secrets.token_urlsafe(24), 'exp': datetime.now(UTC) + timedelta(minutes=10)}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    # This is deliberately a top-level navigation, not a fetch call. A cookie
    # set from a cross-site fetch can be blocked before Google returns here.
    response = RedirectResponse(url=google_authorization_url(state), status_code=303)
    clear_oauth_state_cookie(response)
    response.set_cookie(
        key=OAUTH_STATE_COOKIE_NAME,
        value=state,
        max_age=600,
        httponly=True,
        secure=secure_cookie_setting(settings),
        # Google returns through a top-level GET, for which Lax is sufficient.
        samesite='lax',
        domain=oauth_cookie_domain(),
        path="/",
    )
    return response

@router.get('/auth/google/callback')
def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    oauth_state: str | None = Cookie(default=None, alias=OAUTH_STATE_COOKIE_NAME),
    db: Session = Depends(get_db),
):
    from app.core.config import get_settings
    import jwt

    settings = get_settings()
    frontend_origin = next(
        (origin.strip() for origin in settings.frontend_origins.split(',') if origin.strip()),
        None,
    )
    if not frontend_origin:
        raise HTTPException(503, 'Frontend origin is not configured')

    def callback_failure(message: str):
        response = RedirectResponse(
            url=f"{frontend_origin}/auth/google/callback?{urlencode({'error': message})}",
            status_code=303,
        )
        clear_oauth_state_cookie(response)
        return response

    if error:
        return callback_failure(error_description or error)
    if not state or not code or not oauth_state:
        return callback_failure('Google authentication could not be verified.')

    try:
        state_claims = jwt.decode(
            state,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
        )
        if not state_claims.get('nonce'):
            raise jwt.InvalidTokenError()
    except jwt.InvalidTokenError:
        return callback_failure('Google authentication could not be verified.')

    if not hmac.compare_digest(state, oauth_state):
        return callback_failure('Google authentication could not be verified.')

    try:
        claims = exchange_google_code(code)
    except HTTPException as exc:
        return callback_failure(str(exc.detail))
    except Exception:
        log.exception('Google OAuth exchange failed')
        return callback_failure('Google authentication failed')

    identity = db.scalar(
        select(OAuthIdentity).where(
            OAuthIdentity.provider == 'google',
            OAuthIdentity.subject == claims['sub'],
        )
    )

    if identity:
        user = db.get(User, identity.user_id)
        if not user:
            raise HTTPException(401, 'Google account is not linked to an active user')
    else:
        user = db.scalar(
            select(User).where(
                User.email == claims['email'].lower()
            )
        )

        if not user:
            user = User(
                email=claims['email'].lower(),
                full_name=claims.get('name'),
                password_hash=hash_password(
                    secrets.token_urlsafe(32)
                ),
                is_active=True,
                is_email_verified=True,
            )
            db.add(user)
            db.flush()

        user.is_active = True
        user.is_email_verified = True

        db.add(
            OAuthIdentity(
                user_id=user.id,
                provider='google',
                subject=claims['sub'],
            )
        )

        db.commit()
        db.refresh(user)

    access_token = create_session(db, user)

    # The Vercel app and Render API are cross-site. Some browsers (notably in
    # Incognito) block the API session cookie on the subsequent XHR to /auth/me.
    # Password login already uses a short-lived client-held bearer token, so use
    # the same transport for OAuth as a compatibility handoff. The token is in
    # the fragment, which is not sent in HTTP requests, and the callback page
    # removes it from browser history immediately. The HttpOnly cookie remains
    # available for browsers that permit credentialed cross-site requests.
    response = RedirectResponse(
        url=f"{frontend_origin}/auth/google/callback#{urlencode({'access_token': access_token})}",
        status_code=303,
    )

    secure_cookie = secure_cookie_setting(settings)

    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        max_age=settings.access_token_minutes * 60,
        httponly=True,
        secure=secure_cookie,
        samesite=cookie_samesite_setting(settings),
        domain=oauth_cookie_domain(),
        path="/",
    )

    clear_oauth_state_cookie(response)

    return response

@router.get('/users/me', response_model=UserRead)
def me(user: User = Depends(current_user)): return user

@router.patch('/users/me', response_model=UserRead)
def update_me(body: ProfileUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    user.full_name = body.full_name
    db.commit(); db.refresh(user)
    return user

@router.get('/settings', response_model=SettingsRead)
def get_user_settings(db: Session = Depends(get_db), user: User = Depends(current_user)):
    settings = db.scalar(select(UserSettings).where(UserSettings.user_id == user.id))
    if settings is None:
        settings = UserSettings(user_id=user.id)
        db.add(settings); db.commit(); db.refresh(settings)
    return settings

@router.patch('/settings', response_model=SettingsRead)
def update_user_settings(body: SettingsUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    settings = db.scalar(select(UserSettings).where(UserSettings.user_id == user.id))
    if settings is None:
        settings = UserSettings(user_id=user.id)
        db.add(settings)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)
    db.commit(); db.refresh(settings)
    return settings

@router.get('/conversations', response_model=list[ConversationRead])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(current_user)): return conversations.list(db, user.id)
@router.post('/conversations', response_model=ConversationRead, status_code=201)
def create_conversation(body: ConversationCreate, db: Session = Depends(get_db), user: User = Depends(current_user)): return conversations.create(db, Conversation(user_id=user.id, title=body.title))
@router.get('/conversations/{conversation_id}', response_model=ConversationRead)
def get_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)): return conversations.get(db, user.id, conversation_id)
@router.patch('/conversations/{conversation_id}', response_model=ConversationRead)
def update_conversation(conversation_id: str, body: ConversationUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = conversations.get(db, user.id, conversation_id)
    changes = body.model_dump(exclude_unset=True)
    if 'selected_model_id' in changes and changes['selected_model_id'] is not None:
        model = db.scalar(select(ModelRegistry).where(ModelRegistry.id == changes['selected_model_id'], ModelRegistry.is_active.is_(True)))
        if model is None:
            raise HTTPException(400, 'Model not available')
        provider_config = db.scalar(select(ProviderConfiguration).where(ProviderConfiguration.user_id == user.id, ProviderConfiguration.provider == model.provider))
        if provider_config is None:
            raise HTTPException(400, 'Provider not configured')
        if not provider_config.is_enabled:
            raise HTTPException(400, 'Provider not enabled')
        if model.provider not in {'ollama', 'kie'} and not provider_config.encrypted_api_key:
            raise HTTPException(400, 'Provider credentials unavailable')
    for field, value in changes.items(): setattr(item, field, value)
    db.commit(); db.refresh(item); return item

@router.delete('/conversations/{conversation_id}', status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = conversations.get(db, user.id, conversation_id)
    db.delete(item)
    db.commit()

@router.get('/conversations/{conversation_id}/messages', response_model=list[MessageRead])
def list_messages(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    conversations.get(db, user.id, conversation_id); return list(db.scalars(select(Message).where(Message.user_id == user.id, Message.conversation_id == conversation_id)))
@router.post('/conversations/{conversation_id}/messages', response_model=MessageRead, status_code=201)
def create_message(conversation_id: str, body: MessageCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    conversations.get(db, user.id, conversation_id); return messages.create(db, Message(user_id=user.id, conversation_id=conversation_id, **body.model_dump()))

@router.post('/conversations/{conversation_id}/extract-memories', response_model=list[MemoryRead])
def extract_memories(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    conversations.get(db, user.id, conversation_id)
    return extract_from_conversation(db, user.id, conversation_id)

# ---------------------------------------------------------------------------
# Phase 5 — LLM generation endpoint
# ---------------------------------------------------------------------------

@router.post('/conversations/{conversation_id}/generate')
def generate(
    conversation_id: str,
    body: GenerateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    """Stream an LLM response into the conversation.

    Pipeline:
    1. Verify conversation ownership.
    2. Verify the user has an *enabled* ProviderConfiguration for body.provider.
    3. Verify body.model_key exists in ModelRegistry for body.provider.
    4. Save the user message.
    5. Build context (relevant memories + recent history).
    6. Decrypt key immediately before calling the provider.
    7. Stream response chunks as newline-delimited JSON.
    8. On completion: save assistant message, run memory extraction.
    9. On error: emit safe JSON error chunk — never expose the key.
    """
    # 1. Conversation ownership
    conversations.get(db, user.id, conversation_id)

    # 2. Provider config — must exist and be enabled
    config = db.scalar(
        select(ProviderConfiguration).where(
            ProviderConfiguration.user_id == user.id,
            ProviderConfiguration.provider == body.provider,
        )
    )
    if config is None:
        raise HTTPException(400, 'Provider not configured')
    if not config.is_enabled:
        raise HTTPException(400, 'Provider not enabled')

    # 3. Model must exist in registry for this provider
    # Ollama models are discovered locally and are intentionally not routed
    # through the Render backend. Local generation uses the browser connector.
    model_entry = db.scalar(
        select(ModelRegistry).where(
            ModelRegistry.provider == body.provider,
            ModelRegistry.model_key == body.model_key,
            ModelRegistry.is_active.is_(True),
        )
    )

    # Live-provider models may not exist in the static registry. For configured
    # cloud providers, allow an exact model returned by the provider's own
    # discovery endpoint and materialize it into ModelRegistry for future use.
    if model_entry is None and body.provider not in {'ollama', 'kie'}:
        provider = get_provider(body.provider)
        discovery_key = decrypt_api_key(config.encrypted_api_key) if config.encrypted_api_key else None
        try:
            discovered = provider.list_models(discovery_key)
        except HTTPException:
            raise
        if not any(str(item.get('model_key')) == body.model_key for item in discovered):
            raise HTTPException(400, 'Model not available from provider')
        model_entry = ModelRegistry(
            provider=body.provider,
            model_key=body.model_key,
            display_name=next(
                (str(item.get('display_name') or body.model_key) for item in discovered if str(item.get('model_key')) == body.model_key),
                body.model_key,
            ),
            is_local=False,
            is_active=True,
        )
        db.add(model_entry)
        db.flush()

    if model_entry is None:
        raise HTTPException(400, 'Model not available')

    # 4. Save user message
    user_msg = messages.create(
        db,
        Message(
            user_id=user.id,
            conversation_id=conversation_id,
            role='user',
            content=body.message,
            provider=body.provider,
            model_id=body.model_key,
        ),
    )
    record_event(db, user.id, 'message_sent', body.provider, body.model_key, model_entry.is_local)
    db.commit()

    # 5. Build context (captures db state before streaming begins)
    user_settings = db.scalar(select(UserSettings).where(UserSettings.user_id == user.id))
    memory_enabled = user_settings is None or user_settings.memory_enabled
    llm_messages = context_builder.build(db, user.id, conversation_id, body.message, memory_enabled=memory_enabled)
    if body.image_url is not None and body.provider == 'kie':
        llm_messages[-1]['image_url'] = str(body.image_url)

    # 6. Capture encrypted key — decryption happens inside the generator
    encrypted_key = config.encrypted_api_key

    provider_name = body.provider
    model_key = body.model_key
    user_id = user.id
    max_output_tokens = model_entry.max_output_tokens
    estimated_context_characters = sum(len(str(message.get('content', ''))) for message in llm_messages)
    log.info(
        "LLM generation request provider=%s model=%s memory_enabled=%s messages=%s context_chars=%s",
        provider_name,
        model_key,
        memory_enabled,
        len(llm_messages),
        estimated_context_characters,
    )

    safe_details = {
        'PROVIDER_AUTH_ERROR': f'{provider_name.capitalize()} API key was rejected.',
        'PROVIDER_BILLING_OR_CREDITS': f'{provider_name.capitalize()} account cannot currently serve this request. Check credits or provider access.',
        'PROVIDER_FORBIDDEN': f'{provider_name.capitalize()} denied access to this request.',
        'MODEL_NOT_FOUND': 'The selected model is unavailable from the provider.',
        'PROVIDER_TIMEOUT': f'{provider_name.capitalize()} timed out while processing this request.',
        'PROVIDER_CONFLICT': f'{provider_name.capitalize()} could not accept this request in its current state.',
        'RATE_LIMITED': 'Provider rate limit reached.',
        'PROVIDER_UNAVAILABLE': f'{provider_name.capitalize()} temporarily unavailable.',
        'PROVIDER_ERROR': 'The provider could not complete this request.',
    }

    def normalized_provider_error(exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        status_code = exc.status_code if isinstance(exc, HTTPException) else None
        return provider_error(exc, provider_name, model_key, status_code)

    def event_stream():
        accumulated = []
        try:
            provider = get_provider(provider_name)
            if not encrypted_key and provider_name not in {'ollama', 'kie'}:
                yield json.dumps({'type': 'error', 'code': 'PROVIDER_KEY_MISSING', 'detail': 'API key not configured for this provider. Open Provider Settings and add your key.', 'provider': provider_name, 'model_key': model_key, 'recoverable': False, 'partial': False}) + '\n'
                return
            # Decrypt immediately before use; result is local to this generator
            api_key = decrypt_api_key(encrypted_key) if encrypted_key else None
            for chunk in provider.stream(llm_messages, api_key, model_key, max_output_tokens):
                accumulated.append(chunk)
                yield json.dumps({'type': 'chunk', 'text': chunk}) + '\n'
        except HTTPException as exc:
            normalized = normalized_provider_error(exc)
            log.error(
                "%s generation failed provider=%s model=%s status=%s error_type=%s",
                provider_name.capitalize(),
                provider_name,
                model_key,
                normalized.provider_status,
                normalized.error_type,
            )
            record_event(db, user_id, 'message_failed', provider_name, model_key, model_entry.is_local)
            db.commit()
            yield json.dumps({
                'type': 'error',
                'code': normalized.code,
                'detail': safe_details.get(normalized.code, safe_details['PROVIDER_ERROR']),
                'provider': provider_name,
                'model_key': model_key,
                'recoverable': True,
                'partial': bool(accumulated),
            }) + '\n'
            return
        except Exception as exc:
            log.exception("Unexpected error during streaming for provider=%s model=%s", provider_name, model_key)
            yield json.dumps({
                'type': 'error',
                'code': 'INTERNAL_ERROR',
                'detail': 'The selected model could not complete this request.',
                'provider': provider_name,
                'model_key': model_key,
                'recoverable': True,
                'partial': bool(accumulated),
            }) + '\n'
            return

        # Save completed assistant message and run extraction on success
        final_text = ''.join(accumulated)
        if final_text:
            try:
                asst_msg = Message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role='assistant',
                    content=final_text,
                    provider=provider_name,
                    model_id=model_key,
                )
                db.add(asst_msg)
                db.commit()
                record_event(db, user_id, 'message_completed', provider_name, model_key, model_entry.is_local)
                db.commit()
                db.refresh(asst_msg)
                # Run existing memory extraction heuristic
                extract_from_conversation(db, user_id, conversation_id)
                yield json.dumps({'type': 'done', 'message_id': asst_msg.id}) + '\n'
            except Exception:
                log.exception("Failed to persist assistant message")
                yield json.dumps({'type': 'error', 'detail': 'Failed to save response'}) + '\n'

    return StreamingResponse(event_stream(), media_type='application/x-ndjson')


# ---------------------------------------------------------------------------
# Provider management
# ---------------------------------------------------------------------------

@router.get('/memories', response_model=list[MemoryRead])
def list_memories(db: Session = Depends(get_db), user: User = Depends(current_user)): return memories.list(db, user.id)
@router.get('/memory', response_model=list[MemoryRead])
def list_memory_alias(db: Session = Depends(get_db), user: User = Depends(current_user)): return memories.list(db, user.id)
@router.post('/memories', response_model=MemoryRead, status_code=201)
def create_memory(body: MemoryCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if body.source_conversation_id: conversations.get(db, user.id, body.source_conversation_id)
    memory = memories.create(db, Memory(user_id=user.id, **body.model_dump())); vector_store.upsert(memory); return memory
@router.post('/memory', response_model=MemoryRead, status_code=201)
def create_memory_alias(body: MemoryCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return create_memory(body, db, user)
@router.patch('/memories/{memory_id}', response_model=MemoryRead)
def update_memory(memory_id: str, body: MemoryUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = memories.get(db, user.id, memory_id)
    for field, value in body.model_dump(exclude_unset=True).items(): setattr(item, field, value)
    db.commit(); db.refresh(item)
    if item.is_archived: vector_store.delete(item.id)
    else: vector_store.upsert(item)
    return item
@router.patch('/memory/{memory_id}', response_model=MemoryRead)
def update_memory_alias(memory_id: str, body: MemoryUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return update_memory(memory_id, body, db, user)
@router.get('/memory/{memory_id}', response_model=MemoryRead)
def get_memory_alias(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)): return memories.get(db, user.id, memory_id)
@router.delete('/memories/{memory_id}', status_code=204)
def delete_memory(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = memories.get(db, user.id, memory_id); vector_store.delete(item.id); db.delete(item); record_event(db, user.id, 'memory_deleted'); db.commit()
@router.delete('/memory/{memory_id}', status_code=204)
def delete_memory_alias(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return delete_memory(memory_id, db, user)

@router.post('/memories/{memory_id}/archive', response_model=MemoryRead)
def archive_memory(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = memories.get(db, user.id, memory_id); item.is_archived = True; db.commit(); db.refresh(item); vector_store.delete(item.id); return item

@router.post('/memories/{memory_id}/restore', response_model=MemoryRead)
def restore_memory(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = memories.get(db, user.id, memory_id); item.is_archived = False; db.commit(); db.refresh(item); vector_store.upsert(item); return item

@router.post('/memories/{memory_id}/pin', response_model=MemoryRead)
def pin_memory(memory_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = memories.get(db, user.id, memory_id); item.is_pinned = True; db.commit(); db.refresh(item); vector_store.upsert(item); return item

@router.get('/memories/search', response_model=list[MemorySearchResult])
def search_memories(query: str, limit: int = Query(default=8, ge=1, le=20), db: Session = Depends(get_db), user: User = Depends(current_user)):
    return [MemorySearchResult(**MemoryRead.model_validate(memory).model_dump(), score=score) for memory, score in retrieve(db, user.id, query, limit)]

@router.get('/providers', response_model=list[ProviderRead])
def list_providers(db: Session = Depends(get_db), user: User = Depends(current_user)): return providers.list(db, user.id)

@router.post('/providers', response_model=ProviderRead, status_code=201)
def configure_provider(body: ProviderCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    existing = db.scalar(select(ProviderConfiguration).where(ProviderConfiguration.user_id == user.id, ProviderConfiguration.provider == body.provider))
    if existing: raise HTTPException(409, 'Provider already configured')
    return providers.create(db, ProviderConfiguration(user_id=user.id, provider=body.provider, encrypted_api_key=encrypt_api_key(body.api_key) if body.api_key else None, is_enabled=body.is_enabled))

@router.post('/providers/{provider_name}/credentials', response_model=ProviderRead, status_code=201)
def create_provider_credentials(provider_name: str, body: ProviderCredentialCreate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return configure_provider(ProviderCreate(provider=provider_name, api_key=body.api_key, is_enabled=body.is_enabled), db, user)

@router.put('/providers/{provider_name}', response_model=ProviderRead)
def update_provider(provider_name: str, body: ProviderUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Update API key and/or enabled state for an existing provider configuration."""
    config = db.scalar(
        select(ProviderConfiguration).where(
            ProviderConfiguration.user_id == user.id,
            ProviderConfiguration.provider == provider_name,
        )
    )
    if config is None:
        raise HTTPException(404, 'Provider not configured')
    if body.api_key is not None:
        config.encrypted_api_key = encrypt_api_key(body.api_key)
    if body.is_enabled is not None:
        config.is_enabled = body.is_enabled
    db.commit()
    db.refresh(config)
    return config

@router.post('/providers/{provider_name}/test')
def test_provider(provider_name: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    config = db.scalar(select(ProviderConfiguration).where(ProviderConfiguration.user_id == user.id, ProviderConfiguration.provider == provider_name))
    if config is None or (not config.encrypted_api_key and provider_name not in {'ollama', 'kie'}):
        raise HTTPException(400, 'API key not configured for this provider. Open Provider Settings and add your key.')
    provider = get_provider(provider_name)
    api_key = decrypt_api_key(config.encrypted_api_key) if config.encrypted_api_key else None
    provider.validate_credentials(api_key)
    return {'provider': provider_name, 'status': 'connected'}

@router.delete('/providers/{provider_name}', status_code=204)
def delete_provider(provider_name: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Remove a provider configuration for the authenticated user."""
    config = db.scalar(
        select(ProviderConfiguration).where(
            ProviderConfiguration.user_id == user.id,
            ProviderConfiguration.provider == provider_name,
        )
    )
    if config is None:
        raise HTTPException(404, 'Provider not configured')
    db.delete(config)
    db.commit()

@router.delete('/providers/{provider_name}/credentials', status_code=204)
def delete_provider_credentials(provider_name: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return delete_provider(provider_name, db, user)

@router.get('/providers/{provider_name}/models', response_model=list[ProviderModelRead])
def list_provider_models(provider_name: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """List live models available from the provider.

    Requires the authenticated user to have a ProviderConfiguration row for this
    provider (enabled or not).  Decrypts the key immediately before the call;
    never returns it.
    """
    config = db.scalar(
        select(ProviderConfiguration).where(
            ProviderConfiguration.user_id == user.id,
            ProviderConfiguration.provider == provider_name,
        )
    )
    if config is None:
        raise HTTPException(404, 'Provider not configured')
    provider = get_provider(provider_name)
    # Decrypt key only for the duration of this call; never stored or returned
    api_key = decrypt_api_key(config.encrypted_api_key) if config.encrypted_api_key else None
    raw_models = provider.list_models(api_key)
    return [ProviderModelRead(**m) for m in raw_models]

# ---------------------------------------------------------------------------
# GET /models — filtered by the user's enabled ProviderConfiguration rows
# Security: backend is the authoritative gate; frontend filter is UX only.
# ---------------------------------------------------------------------------

@router.get('/models', response_model=list[ModelRead])
def list_models(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Return ModelRegistry rows only for providers the user has enabled."""
    enabled_providers = db.scalars(
        select(ProviderConfiguration.provider).where(
            ProviderConfiguration.user_id == user.id,
            ProviderConfiguration.is_enabled.is_(True),
        )
    ).all()
    if not enabled_providers:
        return []
    return list(
        db.scalars(
            select(ModelRegistry).where(
                ModelRegistry.is_active.is_(True),
                ModelRegistry.provider.in_(enabled_providers),
            )
        )
    )

@router.post('/providers/ollama/local-models', response_model=list[ModelRead])
def register_local_ollama_models(body: list[LocalModelRegisterRequest], db: Session = Depends(get_db), user: User = Depends(current_user)):
    config = db.scalar(select(ProviderConfiguration).where(ProviderConfiguration.user_id == user.id, ProviderConfiguration.provider == 'ollama'))
    if config is None:
        raise HTTPException(400, 'Ollama is not configured')
    if not config.is_enabled:
        raise HTTPException(400, 'Ollama is not enabled')
    result = []
    for item in body:
        existing = db.scalar(select(ModelRegistry).where(ModelRegistry.model_key == item.model_key))
        if existing is None:
            existing = ModelRegistry(provider='ollama', model_key=item.model_key, display_name=item.display_name or item.model_key, is_local=True, is_active=True, supports_streaming=True)
            db.add(existing)
        elif existing.provider != 'ollama':
            raise HTTPException(409, 'Model key belongs to another provider')
        else:
            existing.display_name = item.display_name or existing.display_name or item.model_key
            existing.is_local = True
            existing.is_active = True
        result.append(existing)
    db.commit()
    for item in result: db.refresh(item)
    return result

@router.post('/conversations/{conversation_id}/prepare-local-generation')
def prepare_local_generation(conversation_id: str, body: LocalGenerationPrepareRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    conversations.get(db, user.id, conversation_id)
    config = db.scalar(select(ProviderConfiguration).where(ProviderConfiguration.user_id == user.id, ProviderConfiguration.provider == 'ollama'))
    if config is None:
        raise HTTPException(400, 'Ollama is not configured')
    if not config.is_enabled:
        raise HTTPException(400, 'Ollama is not enabled')
    model_entry = db.scalar(select(ModelRegistry).where(ModelRegistry.provider == 'ollama', ModelRegistry.model_key == body.model_key, ModelRegistry.is_active.is_(True)))
    if model_entry is None:
        raise HTTPException(400, 'Local Ollama model is not registered')
    user_settings = db.scalar(select(UserSettings).where(UserSettings.user_id == user.id))
    memory_enabled = user_settings is None or user_settings.memory_enabled
    llm_messages = context_builder.build(db, user.id, conversation_id, body.message, memory_enabled=memory_enabled)
    user_msg = messages.create(db, Message(user_id=user.id, conversation_id=conversation_id, role='user', content=body.message, provider='ollama', model_id=body.model_key))
    record_event(db, user.id, 'message_sent', 'ollama', body.model_key, True)
    db.commit()
    db.refresh(user_msg)
    return {'message_id': user_msg.id, 'model_key': body.model_key, 'messages': llm_messages}

@router.post('/conversations/{conversation_id}/complete-local-generation')
def complete_local_generation(conversation_id: str, body: LocalGenerationCompleteRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    conversations.get(db, user.id, conversation_id)
    user_msg = db.scalar(select(Message).where(Message.id == body.user_message_id, Message.user_id == user.id, Message.conversation_id == conversation_id, Message.role == 'user'))
    if user_msg is None:
        raise HTTPException(404, 'Local generation message not found')
    existing = db.scalar(select(Message).where(Message.conversation_id == conversation_id, Message.user_id == user.id, Message.role == 'assistant', Message.model_id == user_msg.model_id, Message.content == body.content))
    if existing is not None:
        return {'message_id': existing.id}
    assistant = Message(user_id=user.id, conversation_id=conversation_id, role='assistant', content=body.content, provider='ollama', model_id=user_msg.model_id)
    db.add(assistant)
    record_event(db, user.id, 'message_completed', 'ollama', user_msg.model_id, True)
    db.commit()
    db.refresh(assistant)
    try:
        extract_from_conversation(db, user.id, conversation_id)
    except Exception:
        log.exception('Local Ollama memory extraction failed')
    return {'message_id': assistant.id}

@router.get('/analytics', response_model=AnalyticsRead)
def analytics(db: Session = Depends(get_db), user: User = Depends(current_user)):
    count = lambda model: db.scalar(select(func.count()).select_from(model).where(model.user_id == user.id)) or 0
    return AnalyticsRead(conversations=count(Conversation), messages=count(Message), memories=count(Memory))

@router.get('/analytics/overview', response_model=AnalyticsOverview)
def analytics_overview(days: int | None = Query(default=30, ge=1, le=3650), db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = analytics_since(days)
    def count(model):
        query = select(func.count()).select_from(model).where(model.user_id == user.id)
        if since is not None: query = query.where(model.created_at >= since)
        return db.scalar(query) or 0
    event_query = select(AnalyticsEvent).where(AnalyticsEvent.user_id == user.id, AnalyticsEvent.event_type.in_(['message_completed', 'message_failed']))
    if since is not None: event_query = event_query.where(AnalyticsEvent.created_at >= since)
    events = list(db.scalars(event_query))
    return AnalyticsOverview(total_conversations=count(Conversation), total_messages=count(Message), total_memories=count(Memory), total_ai_requests=len(events), successful_requests=sum(event.event_type == 'message_completed' for event in events), failed_requests=sum(event.event_type == 'message_failed' for event in events))

@router.get('/analytics/activity', response_model=list[AnalyticsPoint])
def analytics_activity(days: int | None = Query(default=30, ge=1, le=3650), db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = analytics_since(days)
    points: dict[str, AnalyticsPoint] = {}
    def point(timestamp):
        key = timestamp.date().isoformat()
        return points.setdefault(key, AnalyticsPoint(date=key))
    for model, field in ((Conversation, 'conversations'), (Message, 'messages'), (Memory, 'memories')):
        query = select(model.created_at).where(model.user_id == user.id)
        if since is not None: query = query.where(model.created_at >= since)
        for timestamp in db.scalars(query): setattr(point(timestamp), field, getattr(point(timestamp), field) + 1)
    query = select(AnalyticsEvent).where(AnalyticsEvent.user_id == user.id)
    if since is not None: query = query.where(AnalyticsEvent.created_at >= since)
    for event in db.scalars(query):
        target = point(event.created_at)
        if event.event_type == 'message_completed': target.successful_requests += 1
        if event.event_type == 'message_failed': target.failed_requests += 1
    return [points[key] for key in sorted(points)]

@router.get('/analytics/memories', response_model=list[AnalyticsItem])
def analytics_memories(days: int | None = Query(default=30, ge=1, le=3650), db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = analytics_since(days)
    query = select(Memory.category, func.count()).where(Memory.user_id == user.id).group_by(Memory.category)
    if since is not None: query = query.where(Memory.created_at >= since)
    return [AnalyticsItem(name=category, count=count) for category, count in db.execute(query)]

@router.get('/analytics/models', response_model=list[AnalyticsItem])
def analytics_models(days: int | None = Query(default=30, ge=1, le=3650), db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = analytics_since(days)
    query = select(AnalyticsEvent.model_key, AnalyticsEvent.provider, AnalyticsEvent.is_local, func.count()).where(AnalyticsEvent.user_id == user.id, AnalyticsEvent.event_type == 'message_completed').group_by(AnalyticsEvent.model_key, AnalyticsEvent.provider, AnalyticsEvent.is_local)
    if since is not None: query = query.where(AnalyticsEvent.created_at >= since)
    return [AnalyticsItem(name=model or 'Unknown model', provider=provider, is_local=is_local, count=count) for model, provider, is_local, count in db.execute(query)]

@router.get('/analytics/providers', response_model=list[AnalyticsItem])
def analytics_providers(days: int | None = Query(default=30, ge=1, le=3650), db: Session = Depends(get_db), user: User = Depends(current_user)):
    since = analytics_since(days)
    query = select(AnalyticsEvent.provider, AnalyticsEvent.is_local, func.count()).where(AnalyticsEvent.user_id == user.id, AnalyticsEvent.event_type == 'message_completed').group_by(AnalyticsEvent.provider, AnalyticsEvent.is_local)
    if since is not None: query = query.where(AnalyticsEvent.created_at >= since)
    return [AnalyticsItem(name=provider or 'Unknown provider', is_local=is_local, count=count) for provider, is_local, count in db.execute(query)]

@router.get('/privacy/export')
def privacy_export(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return export_user_data(db, user)

@router.get('/privacy/export/memories')
def privacy_export_memories(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return export_memories(db, user.id)

@router.get('/privacy/export/conversations')
def privacy_export_conversations(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return export_conversations(db, user.id)
