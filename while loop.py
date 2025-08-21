balance = 1000
user_input = input("press 1 for deposit, press 2 for withdrawl")
while user_input != "exit":
    if user_input == 'D':
        amount=int(input("enter amount to deposit"))
        balance = amount
        print("you have deposited", amount, "Rupees in your account")
    elif user_input == 'w':
        amount=int(input("enter amount to withdraw"))
        if(amount <= balance):
            balance = amount
            print("you have withdraw amount", amount, "Rupees in your account")
        else:
            print("you don't have enough rupees in your account")
    elif user_input == 'c':
        print("current balance", balance)
    else:
        print("invalid input")
        user_input = input("press D for deposit, press A for withdrwal, for checking the balance, exit to quit").upper()








