import { useEffect, useId, useRef } from "react";
import mermaid from "mermaid";
import type { Components } from "react-markdown";

let mermaidInitialized = false;

function initMermaid(): void {
  if (mermaidInitialized) return;
  mermaidInitialized = true;
  mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });
}

function MermaidDiagram({ code }: { code: string }): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const id = useId();

  useEffect(() => {
    if (!containerRef.current || !code.trim()) return;
    initMermaid();
    const el = containerRef.current;
    el.textContent = code;
    mermaid
      .run({ nodes: [el], suppressErrors: true })
      .catch(() => {});
  }, [code]);

  return (
    <div
      ref={containerRef}
      id={`mermaid-${id.replace(/:/g, "")}`}
      className="mermaid flex justify-center [&>svg]:max-w-full"
    />
  );
}

function getCodeChildren(props: { children?: React.ReactNode }): string {
  const { children } = props;
  if (typeof children === "string") return children;
  if (Array.isArray(children) && children.length > 0 && typeof children[0] === "string") {
    return children[0] as string;
  }
  return String(children ?? "");
}

export const markdownComponents: Components = {
  code(props) {
    const { className, ...rest } = props;
    const isMermaid = typeof className === "string" && className.includes("language-mermaid");
    const code = getCodeChildren(props);
    if (isMermaid && code) {
      return <MermaidDiagram code={code} />;
    }
    return <code className={className} {...rest} />;
  },
};
