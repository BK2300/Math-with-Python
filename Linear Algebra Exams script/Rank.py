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
        return sp.Rational(token)  # exact: supports 1/3, -2, 0.5
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


def read_vector(n: int, name: str) -> sp.Matrix:
    while True:
        txt = input(f"Enter vector {name} with {n} entries: ").strip()
        if txt.lower() in {"q", "quit", "back"}:
            raise KeyboardInterrupt
        txt = txt.replace(",", " ")
        parts = [p for p in txt.split() if p.strip() != ""]
        if len(parts) != n:
            print(f"❌ Expected {n} numbers, got {len(parts)}.")
            print("Example: 1, -2, 3, 4/5")
            continue
        try:
            nums = [parse_number(p) for p in parts]
            return sp.Matrix(nums)
        except InputError as e:
            print(f"❌ {e}")


def main() -> None:
    print("=== Rank Tool ===")
    print("Type 'q' to quit.\n")

    while True:
        try:
            print("Choose mode:")
            print("1) Rank of a matrix A")
            print("2) Rank(D) and Rank([D|s]) (append s as last column)")
            print("0) Exit")
            choice = input("Choice: ").strip()

            if choice in {"0", "q", "quit", "exit"}:
                print("Bye!")
                return

            if choice == "1":
                m = read_int("Rows m: ", 1, 50)
                n = read_int("Cols n: ", 1, 50)
                A = read_matrix(m, n, "A")
                print("\nA ="); sp.pprint(A, use_unicode=True)

                r = A.rank()
                rrefA, pivots = A.rref()
                print(f"\nrank(A) = {r}")
                print("Pivot columns:", pivots)
                print("\nRREF(A) ="); sp.pprint(rrefA, use_unicode=True)

            elif choice == "2":
                n_rows = read_int("Number of rows (dimension) n: ", 1, 50)
                k = read_int("Number of columns in D (k): ", 1, 50)

                D = read_matrix(n_rows, k, "D")
                s = read_vector(n_rows, "s")

                print("\nD ="); sp.pprint(D, use_unicode=True)
                print("\ns ="); sp.pprint(s, use_unicode=True)

                Aug = D.row_join(s)
                rD = D.rank()
                rAug = Aug.rank()

                print(f"\nrank(D) = {rD}")
                print(f"rank([D|s]) = {rAug}")

                if rD == rAug:
                    print("✅ ranks are equal → s does NOT increase rank (often means s is in span(D)).")
                else:
                    print("❌ rank increased → s adds a new independent direction (not in span(D)).")

                print("\nRREF(D) ="); sp.pprint(D.rref()[0], use_unicode=True)
                print("\nRREF([D|s]) ="); sp.pprint(Aug.rref()[0], use_unicode=True)

            else:
                print("Invalid choice.")

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
