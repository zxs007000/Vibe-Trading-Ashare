import { useEffect, useState } from "react";

/**
 * Theme toggle — defaults to DARK (terminal aesthetic).
 * Adds/removes .light class on <html> to switch to light mode.
 * Persists choice as "vt-theme" in localStorage.
 */
export function useDarkMode() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("vt-theme");
    if (saved) return saved !== "light";   // anything except "light" → dark
    return true;                            // default dark
  });

  useEffect(() => {
    document.documentElement.classList.toggle("light", !dark);
    localStorage.setItem("vt-theme", dark ? "dark" : "light");
  }, [dark]);

  return { dark, toggle: () => setDark((d) => !d) };
}
