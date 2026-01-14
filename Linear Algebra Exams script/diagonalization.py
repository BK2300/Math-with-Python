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


def read_int(prompt: str, min_value: int = 0, max_value: int = 50) -> int:
    while True:
        s = input(prompt).strip()
        if s.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        # allow 0 and positive ints
        if not s.isdigit():
            print("Please enter a non-negative integer.")
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


def try_diagonalize(A: sp.Matrix):
    """
    Returns (P, D) if diagonalizable, else None.
    """
    try:
        # reals_only=False allows complex eigenvalues if needed
        P, D = A.diagonalize(reals_only=False)
        return sp.simplify(P), sp.simplify(D)
    except Exception:
        return None


def main() -> None:
    print("=== Diagonalization Tool ===")
    print("Goal: write A = P*D*P^{-1} (if A is diagonalizable).")
    print("Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Matrix size n (square, max 10): ", 1, 10)
            A = read_square_matrix(n, "A")

            print("\nA =")
            sp.pprint(A, use_unicode=True)

            print("\nEigenvalues/eigenvectors (summary):")
            for lam, mult, vecs in A.eigenvects():
                print("\n" + "-" * 44)
                print(f"λ = {sp.simplify(lam)}   (multiplicity = {mult})")
                print("eigenvector basis:")
                for v in vecs:
                    sp.pprint(v, use_unicode=True)

            res = try_diagonalize(A)
            if res is None:
                print("\n❌ A is NOT diagonalizable (or SymPy could not diagonalize it).")
                print("Tip: If your exam covers Jordan form, you could use A.jordan_form().")
            else:
                P, D = res
                print("\n✅ Diagonalization found:")
                print("\nP = (columns are eigenvectors)")
                sp.pprint(P, use_unicode=True)

                print("\nD = (diagonal eigenvalue matrix)")
                sp.pprint(D, use_unicode=True)

                # Verification
                Pinv = sp.simplify(P.inv())
                check1 = sp.simplify(Pinv * A * P)
                check2 = sp.simplify(P * D * Pinv)

                print("\nCheck 1: P^{-1} * A * P =")
                sp.pprint(check1, use_unicode=True)

                print("\nCheck 2: P * D * P^{-1} =")
                sp.pprint(check2, use_unicode=True)

                # Optional: compute A^k
                do_pow = input("\nCompute A^k using diagonalization? (y/n): ").strip().lower()
                if do_pow == "y":
                    k = read_int("k (0..50): ", 0, 50)
                    Ak = sp.simplify(P * (D ** k) * Pinv)
                    print(f"\nA^{k} =")
                    sp.pprint(Ak, use_unicode=True)

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
