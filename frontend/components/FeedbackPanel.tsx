"use client";

import { useEffect, useState } from "react";
import {
  AnalyzeResponse,
  checkAdminToken,
  getAdminToken,
  reportWrong,
  sendFeedback,
  setAdminToken,
} from "@/lib/api";

type Props = {
  result: AnalyzeResponse;
};

type Stage =
  | "idle"
  | "thanks_up"
  | "asking_correction"
  | "submitting"
  | "thanks_down"
  | "thanks_verified"
  | "already_reported"
  | "error";

export default function FeedbackPanel({ result }: Props) {
  const [stage, setStage] = useState<Stage>("idle");
  const [comment, setComment] = useState("");
  const [errMsg, setErrMsg] = useState("");
  const [agreeCount, setAgreeCount] = useState<number>(0);
  const [needed, setNeeded] = useState<number>(3);
  const [isAdmin, setIsAdmin] = useState<boolean>(false);
  const [showAdminInput, setShowAdminInput] = useState<boolean>(false);
  const [adminInput, setAdminInputValue] = useState<string>("");
  const [adminCheckMsg, setAdminCheckMsg] = useState<string>("");

  useEffect(() => {
    let alive = true;
    if (getAdminToken()) {
      checkAdminToken().then((ok) => {
        if (alive) setIsAdmin(ok);
      });
    }
    return () => {
      alive = false;
    };
  }, []);

  const onSaveAdminToken = async () => {
    setAdminToken(adminInput.trim());
    const ok = await checkAdminToken();
    setIsAdmin(ok);
    setAdminCheckMsg(ok ? "Maintainer mode active." : "Token rejected.");
    if (ok) setShowAdminInput(false);
  };

  const onClearAdminToken = () => {
    setAdminToken("");
    setIsAdmin(false);
    setAdminCheckMsg("Signed out.");
  };

  const systemVerdict = {
    verdict: result.verdict,
    verdict_label: result.verdict_label,
    score: result.score,
    p_ai: result.p_ai,
    model_versions: result.model_versions,
  };

  const onThumbsUp = async () => {
    setStage("submitting");
    try {
      await sendFeedback({
        request_id: result.request_id,
        rating: "up",
        system_verdict: systemVerdict,
      });
      setStage("thanks_up");
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Failed to submit");
      setStage("error");
    }
  };

  const onThumbsDown = () => {
    setStage("asking_correction");
  };

  const onSubmitCorrection = async (userVerdict: "real" | "ai") => {
    setStage("submitting");
    try {
      const res = await reportWrong({
        request_id: result.request_id,
        user_verdict: userVerdict,
        filename: result.filename,
        file_size: result.bytes,
        system_verdict: systemVerdict,
        comment: comment.trim(),
      });
      const r = res as unknown as {
        status?: string;
        agree_count?: number;
        needed_for_consensus?: number;
      };
      if (r.status === "already_reported") {
        setStage("already_reported");
        return;
      }
      if (r.review_status === "verified" || r.status === "verified") {
        setStage("thanks_verified");
        return;
      }
      setAgreeCount(r.agree_count ?? 1);
      setNeeded(r.needed_for_consensus ?? 2);
      setStage("thanks_down");
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Failed to submit");
      setStage("error");
    }
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-white/60">
          Was this verdict right?
        </h3>
        {isAdmin ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300 ring-1 ring-amber-500/40">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            Maintainer
          </span>
        ) : (
          <span className="text-[10px] uppercase tracking-wider text-white/30">
            reviewed before training
          </span>
        )}
      </div>

      {isAdmin && stage === "idle" && (
        <p className="mb-3 rounded-md border border-amber-500/20 bg-amber-500/5 p-2 text-xs text-amber-200/80">
          Your corrections are taken as ground truth — they enter the
          training set immediately, no consensus needed.
        </p>
      )}

      {stage === "idle" && (
        <div className="flex items-center gap-2">
          <button
            onClick={onThumbsUp}
            className="flex-1 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-sm font-medium text-emerald-300 transition hover:bg-emerald-500/20"
          >
            👍 Yes, looks correct
          </button>
          <button
            onClick={onThumbsDown}
            className="flex-1 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-2.5 text-sm font-medium text-rose-300 transition hover:bg-rose-500/20"
          >
            👎 No, this is wrong
          </button>
        </div>
      )}

      {stage === "submitting" && (
        <p className="text-sm text-white/60">Submitting…</p>
      )}

      {stage === "thanks_up" && (
        <div className="space-y-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
          <p>Thanks — confirmation logged.</p>
          <p className="text-xs text-emerald-200/70">
            Single ratings don&apos;t change predictions. They&apos;re aggregated
            and reviewed before being added to the next training cycle.
          </p>
        </div>
      )}

      {stage === "asking_correction" && (
        <div className="space-y-3">
          <p className="text-sm text-white/70">
            {isAdmin ? (
              <>
                Mark the canonical label. As maintainer, your verdict is
                trusted as ground truth and lands in the training set
                immediately.
              </>
            ) : (
              <>
                Tell us what it actually is. Your correction goes into a
                review queue — it does <em>not</em> change predictions
                immediately, and is only added to training data after
                either three independent users agree or a maintainer
                manually verifies it.
              </>
            )}
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => onSubmitCorrection("real")}
              className="flex-1 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-200 transition hover:bg-cyan-500/20"
            >
              It&apos;s a real photo
            </button>
            <button
              onClick={() => onSubmitCorrection("ai")}
              className="flex-1 rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/10 px-3 py-2 text-sm font-medium text-fuchsia-200 transition hover:bg-fuchsia-500/20"
            >
              It&apos;s AI-generated
            </button>
          </div>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Optional: which generator? what tipped you off? (max 500 chars)"
            maxLength={500}
            rows={2}
            className="w-full rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-white/80 placeholder:text-white/30 focus:border-white/30 focus:outline-none"
          />
          <button
            onClick={() => setStage("idle")}
            className="text-xs text-white/40 hover:text-white/60"
          >
            Cancel
          </button>
        </div>
      )}

      {stage === "thanks_down" && (
        <div className="space-y-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
          <p>Logged — thank you.</p>
          <p className="text-xs text-rose-200/80">
            {agreeCount === 1
              ? "You're the first to flag this image. We need 2 more independent users to agree (or a maintainer to verify) before it enters the training set."
              : agreeCount >= 3
                ? `${agreeCount} users agree this is wrong. Marked for review — will enter the next training cycle.`
                : `${agreeCount} users have agreed so far. ${needed} more needed for automatic consensus.`}
          </p>
        </div>
      )}

      {stage === "thanks_verified" && (
        <div className="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          <p className="flex items-center gap-1.5 font-medium">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M20 6L9 17l-5-5" />
            </svg>
            Verified by maintainer.
          </p>
          <p className="text-xs text-amber-200/80">
            Saved as ground truth. This image will be added to the next
            training cycle&apos;s hard-negative set.
          </p>
        </div>
      )}

      {stage === "already_reported" && (
        <div className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/70">
          You&apos;ve already reported this image with the same correction.
        </div>
      )}

      {/* Maintainer sign-in / sign-out — small footer always visible */}
      <div className="mt-4 border-t border-white/5 pt-3">
        {isAdmin ? (
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-amber-300/70">
              Signed in as maintainer
            </span>
            <button
              onClick={onClearAdminToken}
              className="text-[10px] uppercase tracking-wider text-white/40 hover:text-white/70"
            >
              Sign out
            </button>
          </div>
        ) : showAdminInput ? (
          <div className="space-y-2">
            <input
              type="password"
              value={adminInput}
              onChange={(e) => setAdminInputValue(e.target.value)}
              placeholder="Admin token"
              className="w-full rounded border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-white/80 placeholder:text-white/30 focus:border-white/30 focus:outline-none"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={onSaveAdminToken}
                className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] font-medium text-amber-200 hover:bg-amber-500/20"
              >
                Sign in
              </button>
              <button
                onClick={() => {
                  setShowAdminInput(false);
                  setAdminCheckMsg("");
                }}
                className="text-[11px] text-white/40 hover:text-white/60"
              >
                Cancel
              </button>
              {adminCheckMsg && (
                <span className="text-[10px] text-white/50">{adminCheckMsg}</span>
              )}
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowAdminInput(true)}
            className="text-[10px] uppercase tracking-wider text-white/30 hover:text-white/60"
          >
            Maintainer sign-in
          </button>
        )}
      </div>

      {stage === "error" && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
          Could not submit: {errMsg}.{" "}
          <button
            onClick={() => setStage("idle")}
            className="underline hover:text-amber-100"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
