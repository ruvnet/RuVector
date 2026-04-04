import { useEffect, useRef } from 'react';
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import 'highlight.js/styles/github-dark.css';

hljs.registerLanguage('javascript', javascript);

interface CodeViewerProps {
  code: string;
  language?: string;
  maxHeight?: string;
  showLineNumbers?: boolean;
}

export function CodeViewer({
  code,
  language = 'javascript',
  maxHeight = '600px',
  showLineNumbers = true,
}: CodeViewerProps) {
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current) {
      codeRef.current.textContent = code;
      hljs.highlightElement(codeRef.current);
    }
  }, [code, language]);

  const lines = code.split('\n');

  return (
    <div
      className="relative rounded-lg border border-surface-600 overflow-hidden"
      style={{ maxHeight }}
    >
      <div className="overflow-auto h-full">
        <div className="flex">
          {showLineNumbers && (
            <div className="flex-shrink-0 select-none text-right pr-3 pl-3 py-3 text-gray-600 font-mono text-xs leading-5 border-r border-surface-600 bg-surface-900 sticky left-0">
              {lines.map((_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>
          )}
          <pre className="flex-1 p-3 overflow-x-auto bg-surface-900">
            <code
              ref={codeRef}
              className={`language-${language} text-xs leading-5 font-mono`}
            >
              {code}
            </code>
          </pre>
        </div>
      </div>
    </div>
  );
}
