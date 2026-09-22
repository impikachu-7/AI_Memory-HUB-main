import { useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/contexts/AuthContext";
import { setAccessToken } from "@/services/api";
import { toast } from "sonner";

export default function GoogleCallbackPage() {
  const [, navigate] = useLocation();
  const { isLoading, refresh } = useAuth();
  const hasProcessed = useRef(false);

  useEffect(() => {
    if (isLoading || hasProcessed.current) return;
    hasProcessed.current = true;
    const error = new URLSearchParams(window.location.search).get("error");
    if (error) {
      setAccessToken(null);
      toast.error(error);
      navigate("/login");
      return;
    }
    // The backend puts the OAuth session token in the fragment rather than the
    // query string: fragments are never sent to servers or included in Referer.
    // This mirrors the existing email/password token transport and survives
    // browsers that block Render's cross-site session cookie in Incognito.
    const accessToken = new URLSearchParams(window.location.hash.slice(1)).get("access_token");
    if (accessToken) {
      setAccessToken(accessToken);
      window.history.replaceState(null, "", window.location.pathname);
    }
    refresh().then((user) => {
      if (user) {
        navigate("/chat");
        return;
      }
        toast.error("Google authentication could not be completed.");
        setAccessToken(null);
        navigate("/login");
      });
  }, [isLoading, navigate, refresh]);

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <div className="text-lg font-semibold">Signing you in...</div>
        <p className="mt-2 text-sm text-muted-foreground">
          Completing Google authentication.
        </p>
      </div>
    </div>
  );
}
