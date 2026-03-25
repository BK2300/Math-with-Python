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
        return sp.Rational(token)  # exact: supports -3, 1/2, 0.5
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


def read_int(prompt: str, min_value: int = 1, max_value: int = 10) -> int:
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


def read_square_matrix(n: int, name: str = "T") -> sp.Matrix:
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
                print("Example row: 1, 0, 0, 5")
    return sp.Matrix(rows)


def is_identity(M: sp.Matrix) -> bool:
    n = M.rows
    return M == sp.eye(n)


def main() -> None:
    print("=== Inverse Transformation Tool ===")
    print("Computes T^{-1} for a square matrix T.")
    print("Also shows Gauss-Jordan result from [T|I] -> [I|T^{-1}] when possible.")
    print("Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Matrix size n (square): ", 1, 10)
            T = read_square_matrix(n, "T")

            print("\nT =")
            sp.pprint(T, use_unicode=True)

            detT = sp.simplify(T.det())
            print(f"\ndet(T) = {detT}")
            if detT == 0:
                print("❌ Not invertible (det(T) = 0).")
            else:
                print("✅ Invertible (det(T) != 0).")

                # Method 1: direct inverse
                T_inv = sp.simplify(T.inv())
                print("\nT^{-1} (direct .inv()) =")
                sp.pprint(T_inv, use_unicode=True)

                # Method 2: Gauss-Jordan via RREF on augmented matrix [T|I]
                I = sp.eye(n)
                Aug = T.row_join(I)
                rrefAug, pivots = Aug.rref()

                left = rrefAug[:, :n]
                right = rrefAug[:, n:]

                print("\nRREF([T|I]) =")
                sp.pprint(rrefAug, use_unicode=True)

                if is_identity(left):
                    print("\n✅ Left block became I, so the right block is T^{-1}:")
                    sp.pprint(sp.simplify(right), use_unicode=True)
                else:
                    print("\n⚠️ Left block did NOT become I (unexpected if det!=0).")
                    print("This can happen in rare cases with symbolic weirdness; direct inverse above is the reference.")

                # Quick verification
                check = sp.simplify(T * T_inv)
                print("\nCheck: T * T^{-1} =")
                sp.pprint(check, use_unicode=True)

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
