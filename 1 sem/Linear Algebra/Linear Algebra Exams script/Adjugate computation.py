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
        return sp.Rational(token)  # supports -3, 1/2, 0.5
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    s = row_text.strip()
    for ch in "[](){}":
        s = s.replace(ch, "")
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) != n:
        raise InputError(f"Expected {n} numbers, got {len(parts)}.")
    return [parse_number(p) for p in parts]


def read_int(prompt: str, min_value: int = 1, max_value: int = 8) -> int:
    # adjugate via cofactors gets heavy fast; 8 is already a lot
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


def read_square_matrix(n: int, name: str = "A") -> sp.Matrix:
    print(f"\nEnter matrix {name} row by row ({n} x {n}).")
    print("Separate numbers with spaces or commas. Fractions allowed (e.g. 1/3).")
    rows = []
    for i in range(n):
        while True:
            txt = input(f"{name} Row {i+1} (n={n}): ")
            if txt.lower() in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 0, -2, 3/4")
    return sp.Matrix(rows)


def cofactor_matrix(A: sp.Matrix) -> sp.Matrix:
    n = A.rows
    C = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            # Minor M_ij: delete row i and col j
            M = A.minor_submatrix(i, j)
            C[i, j] = sp.simplify(((-1) ** (i + j)) * M.det())
    return C


def main() -> None:
    print("=== Adjugate Tool ===")
    print("Computes cofactor matrix C and adjugate adj(A) = C^T.")
    print("Also verifies: A*adj(A) = det(A)*I")
    print("Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Matrix size n (square): ", 1, 8)
            A = read_square_matrix(n, "A")

            print("\nA =")
            sp.pprint(A, use_unicode=True)

            detA = sp.simplify(A.det())
            print(f"\ndet(A) = {detA}")

            C = cofactor_matrix(A)
            adjA = sp.simplify(C.T)

            print("\nCofactor matrix C =")
            sp.pprint(C, use_unicode=True)

            print("\nAdjugate adj(A) = C^T =")
            sp.pprint(adjA, use_unicode=True)

            # Verification identity
            I = sp.eye(n)
            left = sp.simplify(A * adjA)
            right = sp.simplify(detA * I)

            print("\nCheck: A * adj(A) =")
            sp.pprint(left, use_unicode=True)

            print("\nShould equal det(A) * I =")
            sp.pprint(right, use_unicode=True)

            if detA != 0:
                invA = sp.simplify(adjA / detA)
                print("\nSince det(A) != 0, inverse via adjugate is:")
                print("A^{-1} = adj(A) / det(A) =")
                sp.pprint(invA, use_unicode=True)

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
