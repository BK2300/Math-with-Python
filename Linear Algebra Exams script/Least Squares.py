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


def read_int(prompt: str, min_value: int = 1, max_value: int = 200) -> int:
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
    rows = []
    for i in range(m):
        while True:
            txt = input(f"{name} Row {i+1} (n={n}): ")
            if txt.lower() in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def read_vector(m: int, name: str) -> sp.Matrix:
    while True:
        txt = input(f"Enter vector {name} with {m} entries: ").strip()
        if txt.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        txt = txt.replace(",", " ")
        parts = [p for p in txt.split() if p.strip() != ""]
        if len(parts) != m:
            print(f"❌ Expected {m} numbers, got {len(parts)}.")
            print("Example: 1, -2, 3, 4/5")
            continue
        try:
            nums = [parse_number(p) for p in parts]
            return sp.Matrix(nums)
        except InputError as e:
            print(f"❌ {e}")


def vec_norm(v: sp.Matrix) -> sp.Expr:
    return sp.sqrt((v.T * v)[0])


def main() -> None:
    print("=== Least Squares Tool (Ax ≈ b) ===")
    print("Finds x* that minimizes ||Ax - b|| (least squares).")
    print("Shows normal equations: (A^T A)x = A^T b")
    print("Type 'q' to quit.\n")

    while True:
        try:
            m = read_int("Number of rows m (data points): ", 1, 200)
            n = read_int("Number of columns n (unknowns): ", 1, 50)

            if m < n:
                print("\n⚠️ Note: m < n (underdetermined). Least squares still exists but is not the usual case.")
                print("This tool still works, but you may get infinitely many solutions.\n")

            A = read_matrix(m, n, "A")
            b = read_vector(m, "b")

            print("\nA ="); sp.pprint(A, use_unicode=True)
            print("\nb ="); sp.pprint(b, use_unicode=True)

            AtA = sp.simplify(A.T * A)
            Atb = sp.simplify(A.T * b)

            print("\nNormal equations:")
            print("A^T A ="); sp.pprint(AtA, use_unicode=True)
            print("A^T b ="); sp.pprint(Atb, use_unicode=True)

            # Solve normal equations
            # linsolve gives solution set; if unique, it's a single tuple.
            x_syms = sp.symbols(f"x1:{n+1}")
            sol = sp.linsolve((AtA, Atb), *x_syms)

            print("\nSolution set for x (from normal equations):")
            sp.pprint(sol, use_unicode=True)

            # Try to extract one solution (common case: unique)
            x_star = None
            try:
                tup = next(iter(sol))
                x_star = sp.Matrix(list(tup))
            except Exception:
                pass

            if x_star is not None:
                Ax = sp.simplify(A * x_star)
                r = sp.simplify(b - Ax)
                print("\nOne least squares solution x* =")
                sp.pprint(x_star, use_unicode=True)

                print("\nAx* ="); sp.pprint(Ax, use_unicode=True)
                print("\nResidual r = b - Ax* ="); sp.pprint(r, use_unicode=True)

                rn = sp.simplify(vec_norm(r))
                print(f"\n||r|| = {rn}   (≈ {sp.N(rn)})")

                # Optional: numeric approximation for x*
                print("\nNumeric x* ≈")
                sp.pprint(sp.N(x_star), use_unicode=True)

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
