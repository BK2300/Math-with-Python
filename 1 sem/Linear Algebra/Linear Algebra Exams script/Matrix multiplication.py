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


def parse_number(token: str) -> sp.Rational:
    token = token.strip()
    if token == "":
        raise InputError("Empty number.")
    try:
        return sp.Rational(token)  # exact; fractions allowed
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    s = row_text.strip().replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) != n:
        raise InputError(f"Expected {n} numbers, got {len(parts)}.")
    return [parse_number(p) for p in parts]


def read_int(prompt: str, min_value: int = 1, max_value: int = 50) -> int:
    while True:
        s = input(prompt).strip()
        if s.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not s.isdigit():
            print("Please enter a positive integer.")
            continue
        v = int(s)
        if v < min_value or v > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return v


def read_matrix(m: int, n: int, name: str) -> sp.Matrix:
    print(f"\nEnter matrix {name} row by row ({m} x {n}).")
    print("Separate numbers with spaces or commas. Fractions allowed (e.g. 1/3).")
    rows: list[list[sp.Rational]] = []
    for i in range(m):
        while True:
            txt = input(f"{name} Row {i+1} (n={n}): ")
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def main() -> None:
    print("=== Matrix Multiplication Tool ===")
    print("Computes C = A * B.")
    print("Fractions allowed. Type 'q' to quit.\n")

    while True:
        try:
            print("Choose dimensions:")
            m = read_int("A rows (m): ", 1, 50)
            n = read_int("A cols (n): ", 1, 50)
            p = read_int("B cols (p): ", 1, 50)

            print(f"\nSo A is {m}x{n} and B must be {n}x{p}.")
            A = read_matrix(m, n, "A")
            B = read_matrix(n, p, "B")

            print("\nA ="); sp.pprint(A, use_unicode=True)
            print("\nB ="); sp.pprint(B, use_unicode=True)

            C = sp.simplify(A * B)

            print("\nC = A*B =")
            sp.pprint(C, use_unicode=True)

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
