age = int(input("Age: "))

if age < 15:
    print("Medicine cannot be given")

elif 15 <= age < 18:
    weight = int(input("Enter weight: "))
    if weight >= 55:
        print("Medicine can be given")
    else:
        print("Medicine cannot be given")

else:  # age >= 18
    print("Medicine can be given")
