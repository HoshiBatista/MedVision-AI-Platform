import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

interface AppLayoutProps {
  title: ReactNode;
  headerExtra?: ReactNode;
  headerActions?: ReactNode;
  children: ReactNode;
}

export function AppLayout({ title, headerExtra, headerActions, children }: AppLayoutProps) {
  return (
    <div id="app">
      <Sidebar />
      <div className="main-layout">
        <header className="main-header">
          <h1>{title}</h1>
          {headerExtra}
          <div style={{ flex: 1 }} />
          {headerActions}
        </header>
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
}
