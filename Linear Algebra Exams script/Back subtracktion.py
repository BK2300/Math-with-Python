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
        val = int(s)
        if val < min_value or val > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return val


def read_augmented_matrix(m: int, n_vars: int) -> sp.Matrix:
    """
    Read an augmented matrix with shape m x (n_vars+1).
    """
    total_cols = n_vars + 1
    print("\nEnter the AUGMENTED matrix row by row (including the last b column).")
    print(f"Each row must have {total_cols} numbers: a1 a2 ... a{n_vars} b")
    print("You can separate numbers with spaces or commas. Fractions allowed (e.g. 1/3).")

    rows: list[list[sp.Rational]] = []
    for i in range(m):
        while True:
            row_text = input(f"Row {i+1} ({total_cols} numbers): ")
            try:
                rows.append(parse_row(row_text, total_cols))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example: 1, 2, -3, 7   (for 3 vars -> 4 numbers)")
    return sp.Matrix(rows)


def is_inconsistent_row(row: sp.Matrix, n_vars: int) -> bool:
    """
    Checks if row looks like [0 0 ... 0 | nonzero]
    """
    left = row[:, :n_vars]
    b = row[0, n_vars]
    return all(left[0, j] == 0 for j in range(n_vars)) and b != 0


def back_substitution(Ab: sp.Matrix, n_vars: int) -> tuple[list[sp.Expr], list[int]]:
    """
    Perform back substitution on an echelon/upper-triangular augmented matrix Ab.
    Returns:
      - solution list x (length n_vars) where free variables are left as symbols
      - list of free variable indices
    Works for:
      - unique solution
      - infinitely many solutions (some free vars)
    Detects inconsistency.
    """
    m, total_cols = Ab.rows, Ab.cols
    if total_cols != n_vars + 1:
        raise InputError("Augmented matrix has wrong number of columns.")

    # Inconsistency check
    for i in range(m):
        if is_inconsistent_row(Ab[i, :], n_vars):
            raise InputError("System is inconsistent (row of form 0=nonzero). No solution.")

    # Determine pivot columns (simple scan for first nonzero in each row)
    pivot_col_for_row = [-1] * m
    pivot_cols = set()
    for i in range(m):
        for j in range(n_vars):
            if Ab[i, j] != 0:
                pivot_col_for_row[i] = j
                pivot_cols.add(j)
                break

    free_cols = [j for j in range(n_vars) if j not in pivot_cols]

    # Create symbols for free variables
    t = sp.symbols(f"t0:{len(free_cols)}")
    x = [sp.Integer(0)] * n_vars
    for k, j in enumerate(free_cols):
        x[j] = t[k]

    # Back substitute from bottom to top
    for i in range(m - 1, -1, -1):
        pc = pivot_col_for_row[i]
        if pc == -1:
            continue  # zero row
        rhs = Ab[i, n_vars]
        # subtract known terms to the right of pivot
        for j in range(pc + 1, n_vars):
            rhs = sp.simplify(rhs - Ab[i, j] * x[j])
        x[pc] = sp.simplify(rhs / Ab[i, pc])

    return x, free_cols


def main() -> None:
    print("=== Back Substitution Tool ===")
    print("Use this after you have an echelon/upper-triangular augmented matrix [A|b].")
    print("Fractions allowed. Type 'q' to quit.\n")

    while True:
        try:
            n_vars = read_int("Number of variables (n): ", 1, 20)
            m = read_int("Number of rows (m): ", 1, 30)

            Ab = read_augmented_matrix(m, n_vars)
            print("\nYou entered [A|b] =")
            sp.pprint(Ab, use_unicode=True)

            sol, free_cols = back_substitution(Ab, n_vars)

            print("\nSolution:")
            for i, expr in enumerate(sol, start=1):
                print(f"x{i} = {expr}")

            if free_cols:
                print("\nFree variables (0-indexed columns):", free_cols)
                print("They are shown as parameters t0, t1, ...")

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except InputError as e:
            print(f"\n⚠️ {e}")
            input("Press Enter to continue...")
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
