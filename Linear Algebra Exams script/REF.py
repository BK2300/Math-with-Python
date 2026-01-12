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
        return sp.Rational(token)  # exact (fractions allowed)
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


def ref(A: sp.Matrix) -> tuple[sp.Matrix, tuple[int, ...]]:
    """
    Compute REF (row echelon form) using Gaussian elimination.
    - We do NOT normalize pivots to 1 (still valid REF).
    - We make entries below pivots zero.
    Returns: (REF_matrix, pivot_columns)
    """
    M = A.copy()
    m, n = M.rows, M.cols
    pivot_cols: list[int] = []

    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break

        # find a non-zero pivot at or below pivot_row
        pivot = None
        for r in range(pivot_row, m):
            if M[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        # swap pivot into place
        if pivot != pivot_row:
            M.row_swap(pivot, pivot_row)

        pivot_val = M[pivot_row, col]
        pivot_cols.append(col)

        # eliminate below pivot
        for r in range(pivot_row + 1, m):
            if M[r, col] == 0:
                continue
            factor = sp.simplify(M[r, col] / pivot_val)

            # IMPORTANT: row_op signature is f(value, column_index)
            M.row_op(r, lambda val, j: sp.simplify(val - factor * M[pivot_row, j]))

        pivot_row += 1

    return M, tuple(pivot_cols)


def main() -> None:
    print("=== REF Tool (Row Echelon Form) ===")
    while True:
        try:
            A = read_matrix()
            print("\nYou entered A =")
            sp.pprint(A, use_unicode=True)

            E, pivots = ref(A)
            print("\nREF(A) =")
            sp.pprint(E, use_unicode=True)
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
