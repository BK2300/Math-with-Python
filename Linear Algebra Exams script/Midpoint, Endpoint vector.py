from __future__ import annotations
import sys

try:
    import sympy as sp
except Exception as e:
    print("Could not import sympy. Install it first with: pip install sympy")
    print("Error:", e)
    sys.exit(1)


class InputError(Exception):
    pass


def parse_num(s: str) -> sp.Rational:
    s = s.strip()
    if s == "":
        raise InputError("Empty input.")
    try:
        return sp.Rational(s)  # supports 1/2, -3, 0.25
    except Exception:
        raise InputError(f"Could not parse '{s}'. Use e.g. 2, -3, 1/2, 0.5")


def read_num(prompt: str) -> sp.Rational:
    while True:
        txt = input(prompt).strip()
        if txt.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            return parse_num(txt)
        except InputError as e:
            print(f"❌ {e}")


def midpoint(x1, y1, x2, y2):
    return sp.simplify((x1 + x2) / 2), sp.simplify((y1 + y2) / 2)


def endpoint_from_midpoint(x1, y1, xmid, ymid):
    # (x1 + x2)/2 = xmid  => x2 = 2*xmid - x1
    # (y1 + y2)/2 = ymid  => y2 = 2*ymid - y1
    return sp.simplify(2 * xmid - x1), sp.simplify(2 * ymid - y1)


def main() -> None:
    print("=== Midpoint / Endpoint Calculator (2D) ===")
    print("Type 'q' anytime to quit.\n")

    while True:
        try:
            print("Choose:")
            print("1) Midpoint (given two endpoints)")
            print("2) Endpoint (given one endpoint + midpoint)")
            choice = input("Choice (1/2): ").strip()

            if choice.lower() in {"q", "quit", "exit"}:
                print("Bye!")
                return

            if choice == "1":
                print("\n--- Midpoint ---")
                x1 = read_num("x1: ")
                y1 = read_num("y1: ")
                x2 = read_num("x2: ")
                y2 = read_num("y2: ")

                xm, ym = midpoint(x1, y1, x2, y2)
                print("\nAnswer:")
                print(f"Midpoint = ({xm}, {ym})")
                print(f"≈ ({sp.N(xm)}, {sp.N(ym)})")

            elif choice == "2":
                print("\n--- Endpoint ---")
                x1 = read_num("x1: ")
                y1 = read_num("y1: ")
                xmid = read_num("xMid: ")
                ymid = read_num("yMid: ")

                x2, y2 = endpoint_from_midpoint(x1, y1, xmid, ymid)
                print("\nAnswer:")
                print(f"Endpoint = ({x2}, {y2})")
                print(f"≈ ({sp.N(x2)}, {sp.N(y2)})")

            else:
                print("Invalid choice. Pick 1 or 2.")
                continue

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            return


if __name__ == "__main__":
    main()
