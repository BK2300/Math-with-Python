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
        return sp.Rational(token)  # exact: supports 1/3, 0.5, -2
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    s = row_text.strip().replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) != n:
        raise InputError(f"Expected {n} numbers, got {len(parts)}.")
    return [parse_number(p) for p in parts]


def read_int(prompt: str, min_value: int = 1, max_value: int = 10) -> int:
    # eigen computations can get slow; keep n reasonable
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


def read_square_matrix() -> sp.Matrix:
    print("\nEnter matrix size (square matrix). Type 'q' to quit.")
    n = read_int("Size n (matrix is n x n): ", 1, 10)

    print("\nEnter the matrix row by row.")
    print("You can separate numbers with spaces or commas.")
    print("Fractions are allowed (e.g. 1/3).")

    rows: list[list[sp.Rational]] = []
    for i in range(n):
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
    print("=== Eigen Tool (Eigenvalues & Eigenvectors) ===")
    print("Definition: A*v = λ*v")
    print("This tool prints eigenvalues λ and a basis for eigenvectors v.")
    print("Fractions are allowed. Type 'q' to quit.\n")

    while True:
        try:
            A = read_square_matrix()
            print("\nYou entered A =")
            sp.pprint(A, use_unicode=True)

            # eigenvects() returns: (eigenvalue, algebraic multiplicity, [eigenvector basis])
            ev = A.eigenvects()

            if not ev:
                print("\nNo eigen information returned (unexpected).")
            else:
                print("\nEigenvalues and eigenvectors:")
                for (lam, mult, vecs) in ev:
                    print("\n" + "-" * 44)
                    print(f"eigenvalue λ = {sp.simplify(lam)}    (multiplicity = {mult})")
                    print("eigenvector basis:")
                    for v in vecs:
                        sp.pprint(v, use_unicode=True)

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
