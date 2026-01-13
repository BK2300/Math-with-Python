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
        return sp.Rational(token)
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


def read_matrix(m: int, n: int) -> sp.Matrix:
    print("\nEnter the matrix row by row.")
    print("Separate numbers with spaces or commas. Fractions allowed (e.g. 1/3).")
    rows: list[list[sp.Rational]] = []
    for i in range(m):
        while True:
            txt = input(f"Row {i+1} (n={n}): ")
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def main() -> None:
    print("=== Linear Independence Tool ===")
    print("We treat the columns of a matrix A as vectors v1, v2, ..., vk.")
    print("They are linearly independent  <=>  only solution to A*c = 0 is c=0.")
    print("Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Vector dimension n (rows): ", 1, 50)
            k = read_int("Number of vectors k (columns): ", 1, 50)

            print(f"\nEnter a matrix A of size {n} x {k}.")
            print("Each COLUMN is one vector.")
            A = read_matrix(n, k)

            print("\nYou entered A =")
            sp.pprint(A, use_unicode=True)

            rrefA, pivots = A.rref()
            rankA = A.rank()

            print("\nRREF(A) =")
            sp.pprint(rrefA, use_unicode=True)
            print("\nPivot columns:", pivots)
            print("rank(A) =", rankA)

            independent = (rankA == k)
            print("\nResult:")
            if independent:
                print("✅ The columns are LINEARLY INDEPENDENT.")
            else:
                print("❌ The columns are LINEARLY DEPENDENT.")

                # Find a non-trivial solution to A*c = 0 (nullspace vector)
                ns = A.nullspace()
                if ns:
                    c = ns[0]  # one dependency vector
                    print("\nOne dependency (non-trivial) coefficient vector c such that A*c = 0:")
                    sp.pprint(c, use_unicode=True)
                    print("\nThis means: c1*v1 + c2*v2 + ... + ck*vk = 0 (not all ci = 0).")
                else:
                    print("\n(Nullspace not returned, but rank < k so dependent.)")

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
