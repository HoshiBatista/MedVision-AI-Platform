import { useToastStore } from "@/store/toast";

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);

  return (
    <div id="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type}`} onClick={() => dismiss(t.id)}>
          <span className="toast-msg">{t.message}</span>
        </div>
      ))}
    </div>
  );
}
