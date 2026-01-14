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


def parse_num(token: str) -> sp.Rational:
    token = token.strip()
    if token == "":
        raise InputError("Empty number.")
    try:
        return sp.Rational(token)  # exact: supports -3, 1/2, 0.5
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


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


def read_vector(n: int, name: str) -> sp.Matrix:
    """
    Accepts: "1, -2, 3" or "1 -2 3"
    Returns n×1 column vector.
    """
    while True:
        txt = input(f"Enter vector {name} with {n} entries: ").strip()
        if txt.lower() in {"q", "quit", "back"}:
            raise KeyboardInterrupt
        txt = txt.replace(",", " ")
        parts = [p for p in txt.split() if p.strip() != ""]
        if len(parts) != n:
            print(f"❌ Expected {n} numbers, got {len(parts)}.")
            print("   Example: 1, -2, 3, 4/5")
            continue
        try:
            nums = [parse_num(p) for p in parts]
            return sp.Matrix(nums)
        except InputError as e:
            print(f"❌ {e}")


def dot(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
    if a.shape != b.shape:
        raise InputError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return (a.T * b)[0]


def norm(v: sp.Matrix) -> sp.Expr:
    return sp.sqrt((v.T * v)[0])


def main() -> None:
    print("=== Orthogonality Tool ===")
    print("Orthogonal means: a·b = 0")
    print("Orthonormal set means: pairwise orthogonal AND each has norm 1")
    print("Type 'q' to quit.\n")

    while True:
        try:
            print("Choose mode:")
            print("1) Check if TWO vectors are orthogonal")
            print("2) Check if a SET of vectors is orthogonal / orthonormal")
            print("0) Exit")
            choice = input("Choice: ").strip().lower()

            if choice in {"0", "q", "quit", "exit"}:
                print("Bye!")
                return

            if choice == "1":
                n = read_int("Vector length n: ", 1, 200)
                a = read_vector(n, "a")
                b = read_vector(n, "b")

                da = sp.simplify(dot(a, b))
                print("\na ="); sp.pprint(a, use_unicode=True)
                print("\nb ="); sp.pprint(b, use_unicode=True)
                print(f"\na·b = {da}")

                if da == 0:
                    print("✅ Orthogonal (a·b = 0).")
                else:
                    print("❌ Not orthogonal (a·b ≠ 0).")

            elif choice == "2":
                n = read_int("Vector length n: ", 1, 200)
                k = read_int("How many vectors? k: ", 1, 50)

                vecs: list[sp.Matrix] = []
                for i in range(1, k + 1):
                    vecs.append(read_vector(n, f"v{i}"))

                # Gram matrix G_ij = v_i · v_j
                G = sp.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        G[i, j] = sp.simplify(dot(vecs[i], vecs[j]))

                print("\nDot product table (Gram matrix G where G[i,j]=vi·vj):")
                sp.pprint(G, use_unicode=True)

                # Orthogonal if off-diagonals are 0
                orthogonal = True
                for i in range(k):
                    for j in range(k):
                        if i != j and G[i, j] != 0:
                            orthogonal = False
                            break
                    if not orthogonal:
                        break

                norms = [sp.simplify(norm(v)) for v in vecs]
                orthonormal = orthogonal and all(ni == 1 for ni in norms)

                print("\nNorms:")
                for i, ni in enumerate(norms, start=1):
                    print(f"||v{i}|| = {ni}  (≈ {sp.N(ni)})")

                if orthogonal:
                    print("\n✅ The set is ORTHOGONAL (all pairwise dot products are 0).")
                else:
                    print("\n❌ The set is NOT orthogonal.")

                if orthonormal:
                    print("✅ The set is ORTHONORMAL (orthogonal + all norms are 1).")
                else:
                    print("❌ The set is NOT orthonormal.")

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
