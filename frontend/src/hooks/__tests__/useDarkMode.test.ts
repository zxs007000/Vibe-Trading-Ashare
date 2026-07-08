import { renderHook, act } from "@testing-library/react";
import { useDarkMode } from "../useDarkMode";

describe("useDarkMode", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  it("defaults to dark when no preference stored", () => {
    const { result } = renderHook(() => useDarkMode());
    expect(result.current.dark).toBe(true);
  });

  it("reads dark preference from localStorage", () => {
    localStorage.setItem("vt-theme", "dark");
    const { result } = renderHook(() => useDarkMode());
    expect(result.current.dark).toBe(true);
  });

  it("reads light preference from localStorage", () => {
    localStorage.setItem("vt-theme", "light");
    const { result } = renderHook(() => useDarkMode());
    expect(result.current.dark).toBe(false);
  });

  it("toggles dark mode", () => {
    const { result } = renderHook(() => useDarkMode());
    expect(result.current.dark).toBe(true);

    act(() => result.current.toggle());
    expect(result.current.dark).toBe(false);

    act(() => result.current.toggle());
    expect(result.current.dark).toBe(true);
  });

  it("persists preference to localStorage on change", () => {
    const { result } = renderHook(() => useDarkMode());
    expect(localStorage.getItem("vt-theme")).toBe("dark");

    act(() => result.current.toggle());
    expect(localStorage.getItem("vt-theme")).toBe("light");

    act(() => result.current.toggle());
    expect(localStorage.getItem("vt-theme")).toBe("dark");
  });

  it("toggles light class on document.documentElement (not dark)", () => {
    const { result } = renderHook(() => useDarkMode());
    // Default: no "light" class
    expect(document.documentElement.classList.contains("light")).toBe(false);

    act(() => result.current.toggle());
    // After toggle: should have "light" class (switched to light mode)
    expect(document.documentElement.classList.contains("light")).toBe(true);

    act(() => result.current.toggle());
    // Toggle back: "light" removed
    expect(document.documentElement.classList.contains("light")).toBe(false);
  });
});
