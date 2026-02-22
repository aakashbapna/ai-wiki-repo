import { Link, NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import WikiPage from "./pages/WikiPage";
import AdminPage from "./pages/AdminPage";

const navLinkClass = ({ isActive }: { isActive: boolean }): string =>
  `text-sm uppercase tracking-wide ${isActive ? "text-accent" : "text-ink/70"}`;

export default function App(): JSX.Element {
  return (
    <div className="min-h-screen">
      <header className="border-b border-ink/10 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <Link to="/" className="group">
            <p className="font-display text-xl font-semibold">Repo Wiki</p>
            <p className="text-xs text-ink/60">Developer-first knowledge hub</p>
          </Link>
          <nav className="flex gap-6 font-display">
            <NavLink to="/" className={navLinkClass}>
              Home
            </NavLink>
            <NavLink to="/admin" className={navLinkClass}>
              Admin
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/wiki/:repoHash" element={<WikiPage />} />
          <Route path="/wiki/:repoHash/page/:pageId" element={<WikiPage />} />
        </Routes>
      </main>
    </div>
  );
}
