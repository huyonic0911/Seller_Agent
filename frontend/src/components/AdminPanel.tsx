import { useEffect, useState } from "react";
import { config } from "../config";

export function AdminPanel({
  onSend,
  ready,
}: {
  onSend: (text: string, author?: string) => void;
  ready: boolean;
}) {
  const [text, setText] = useState("");
  const [author, setAuthor] = useState("khách");
  const [samples, setSamples] = useState<string[]>([]);

  useEffect(() => {
    fetch(`${config.httpBase}/samples`)
      .then((r) => r.json())
      .then((d) => setSamples(d.samples ?? []))
      .catch(() => {});
  }, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (!t) return;
    onSend(t, author.trim() || "khách");
    setText("");
  };

  return (
    <div className="admin-panel">
      <h3>Giả lập comment {ready ? "🟢" : "🔴"}</h3>
      <form onSubmit={submit}>
        <input
          className="author-input"
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder="tên khách"
        />
        <div className="row">
          <input
            className="comment-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Nhập comment của khách..."
          />
          <button type="submit" disabled={!ready}>
            Gửi
          </button>
        </div>
      </form>
      <div className="samples">
        {samples.map((s) => (
          <button key={s} className="sample-chip" onClick={() => onSend(s, author || "khách")} disabled={!ready}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
