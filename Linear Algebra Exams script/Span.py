"""
n = Rows
m = columns

"""

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
            return sp.Matrix(nums)  # column vector
        except InputError as e:
            print(f"❌ {e}")


def main() -> None:
    print("=== Span Membership Tool ===")
    print("Checks whether s is in span(columns of D).")
    print("That means: does D*c = s have a solution?")
    print("Fractions allowed. Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Vector dimension n (rows): ", 1, 50)
            k = read_int("Number of spanning vectors k (columns of D): ", 1, 50)

            print("\nIMPORTANT: The COLUMNS of D are the vectors you want the span of.")
            D = read_matrix(n, k, "D")
            s = read_vector(n, "s")

            print("\nD ="); sp.pprint(D, use_unicode=True)
            print("\ns ="); sp.pprint(s, use_unicode=True)

            Aug = D.row_join(s)
            rank_D = D.rank()
            rank_Aug = Aug.rank()

            print(f"\nrank(D) = {rank_D}")
            print(f"rank([D|s]) = {rank_Aug}")

            if rank_D != rank_Aug:
                print("\n❌ Result: s is NOT in span(D). (System is inconsistent)")
            else:
                print("\n✅ Result: s IS in span(D). (System is consistent)")

                # Solve D*c = s
                c_symbols = sp.symbols(f"c1:{k+1}")
                sol_set = sp.linsolve((D, s), *c_symbols)

                print("\nOne solution set for coefficients c (where D*c = s):")
                sp.pprint(sol_set, use_unicode=True)

                # If you want one concrete solution (pick parameters = 0)
                # SymPy often uses t0, t1, ... for free vars
                try:
                    sol_tuple = next(iter(sol_set))  # one tuple expression
                    free_syms = set().union(*[expr.free_symbols for expr in sol_tuple])
                    # remove the coefficient symbols themselves (shouldn't be in tuple)
                    subs = {sym: 0 for sym in free_syms if str(sym).startswith("t")}
                    particular = [sp.simplify(expr.subs(subs)) for expr in sol_tuple]
                    print("\nA particular solution (setting parameters t0,t1,... = 0):")
                    for i, val in enumerate(particular, start=1):
                        print(f"c{i} = {val}")
                except Exception:
                    pass

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
