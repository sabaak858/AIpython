age = input("age")
citizenship = input("citizenship")

if citizenship == 'true' and age >= 18:
    print("you can vote")
else:
    if citizenship == 'true' and age > 18:
        print("you can not vote")
    else:
        if citizenship == 'flase' and age < 18:
            print("you are not able to vote")


