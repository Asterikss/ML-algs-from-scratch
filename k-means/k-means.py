# pyright: reportUnusedVariable=false
# pyright: ignore - single line (# type: ignore)
# pyright: ignore [reportUnusedVariable=] - single line
class Variables:
    k = 0










def ask_for_k_value():
    Variables.k = int(input("Enter k value"))

def calc_euclidean_distance(a :tuple, b :tuple):
    # print(a)
    # print(len(a))
    if (len(a) != len(b)):
        print("tuples are of different size")
        return -1
    dist = 0;

    for i in range(0 , len(a)):
        dist += (a[i]-b[i])**2

    print(f"c_e_d: {dist**(1/2)}")
    return dist**(1/2)

def main():
    a = (1,3,5)
    b = (3,1,4)
    calc_euclidean_distance(a,b)

if __name__ == "__main__":
    main()
