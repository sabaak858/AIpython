prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
total = 0

while True:
    choice = int(input("Give product number (1-10, 0=quit): "))

    if choice == 0:
        break
    elif 1 <= choice <= 10:
        price = prices[choice - 1]
        total += price
        print("Product:", choice, "Price:", price)
    else:
        print("Incorrect selection.")

print("Total:", total)

payment = int(input("Payment: "))
change = payment - total
print("Change:", change)

