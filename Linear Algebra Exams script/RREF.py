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
        # Exact arithmetic (fractions allowed: 1/3)
        return sp.Rational(token)
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    # Accept "1,2,3" or "1 2 3"
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
        val = int(s)
        if val < min_value or val > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return val


def read_matrix() -> sp.Matrix:
    print("\nEnter matrix size (type 'q' to quit).")
    m = read_int("Number of rows m: ", 1, 30)
    n = read_int("Number of columns n: ", 1, 30)

    print("\nEnter the matrix row by row.")
    print("You can separate numbers with spaces or commas.")
    print("Fractions are allowed (e.g. 1/3).")
    rows: list[list[sp.Rational]] = []
    for i in range(m):
        while True:
            row_text = input(f"Row {i+1} (n={n}): ")
            try:
                row = parse_row(row_text, n)
                rows.append(row)
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def main() -> None:
    print("=== RREF Tool (Row Reduced Echelon Form) ===")
    while True:
        try:
            A = read_matrix()
            print("\nYou entered A =")
            sp.pprint(A, use_unicode=True)

            R, pivots = A.rref()
            print("\nRREF(A) =")
            sp.pprint(R, use_unicode=True)
            print("\nPivot columns:", pivots)

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
