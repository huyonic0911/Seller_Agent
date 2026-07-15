export interface ChatItem {
  id: number;
  kind: "comment" | "reply" | "error";
  author?: string;
  comment?: string;
  text: string;
}

export function ChatOverlay({
  items,
  currentReply,
}: {
  items: ChatItem[];
  currentReply: string | null;
}) {
  return (
    <div className="chat-overlay">
      <div className="chat-list">
        {items.map((it) => (
          <div key={it.id} className={`chat-item ${it.kind}`}>
            {it.kind === "comment" && (
              <span>
                <b>{it.author}:</b> {it.text}
              </span>
            )}
            {it.kind === "reply" && (
              <span>
                <b>🛍️ Shop:</b> {it.text}
              </span>
            )}
            {it.kind === "error" && <span className="err">⚠️ {it.text}</span>}
          </div>
        ))}
      </div>
      {currentReply && <div className="speech-bubble">{currentReply}</div>}
    </div>
  );
}
