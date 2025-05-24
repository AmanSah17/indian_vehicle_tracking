def calculate_sum(a, b):
    return a + b

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        try:
            num1 = float(sys.argv[1])
            num2 = float(sys.argv[2])
            result = calculate_sum(num1, num2)
            print(result)
        except ValueError:
            print("Error: Please provide valid numbers")
    else:
        print("Error: Please provide two numbers as arguments") 