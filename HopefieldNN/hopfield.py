import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import random


class HopfieldAssociativeMemory:
    def __init__(self, input_size, unipolar=True):
        self.input_size = input_size
        self.unipolar = unipolar
        self.weights = np.zeros((input_size, input_size))

    def train(self, input_patterns):
        for pattern in input_patterns:
            pattern = np.array(pattern)
            if not self.unipolar:
                pattern[pattern == 0] = -1
            self.weights += np.outer(pattern, pattern)

        np.fill_diagonal(self.weights, 0)  # Diagonal elements should be 0

    def recall(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern)
        if not self.unipolar:
            input_pattern[input_pattern == 0] = -1

        current_pattern = input_pattern.copy()
        prev_pattern = np.zeros_like(current_pattern)  # Initialize previous pattern
        cycle_detected = False

        for i in range(max_iterations):
            next_pattern = np.sign(np.dot(self.weights, current_pattern))

            # Check for convergence
            if np.array_equal(next_pattern, current_pattern):
                return current_pattern

            # Check for cycle of length two
            if np.array_equal(next_pattern, prev_pattern):
                cycle_detected = True
                break

            prev_pattern = current_pattern.copy()  # Update previous pattern
            current_pattern = next_pattern

        # Handle ending iterations
        if cycle_detected:
            print("Cycle of length two detected.")
            return current_pattern, prev_pattern
        else:
            print("Network response after convergence:")
            return current_pattern


def read_input_from_file(file_path):
    with open(file_path, 'r') as file:
        input_patterns = []
        for line in file:
            pattern = [int(x) for x in line.strip()]
            input_patterns.append(pattern)
    return input_patterns


def read_input_from_gui(input_size):
    def submit():
        nonlocal input_patterns
        pattern_str = pattern_entry.get()
        if len(pattern_str) != input_size:
            messagebox.showerror("Error", f"Pattern length must be {input_size}.")
            return
        pattern = [int(x) for x in pattern_str]
        input_patterns.append(pattern)
        pattern_entry.delete(0, tk.END)
        gui.quit()  # Terminate GUI loop after input is received

    input_patterns = []
    gui = tk.Tk()
    gui.title("Input Patterns")
    pattern_label = tk.Label(gui, text=f"Enter pattern of size {input_size}:")
    pattern_label.pack()
    pattern_entry = tk.Entry(gui, width=input_size)
    pattern_entry.pack()
    submit_button = tk.Button(gui, text="Submit", command=submit)
    submit_button.pack()
    gui.mainloop()
    return input_patterns


def choose_input_method():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    choice = simpledialog.askstring("Input Method", "Choose input method ('file' or 'gui'): ")
    return choice.lower().strip()


def choose_test_pattern(input_patterns):
    choice = simpledialog.askstring("Test Pattern", "Choose test pattern method ('manual' or 'random'): ")
    choice = choice.lower().strip()

    input_size = len(input_patterns[0])  # Get the size of input patterns

    if choice == 'manual':
        while True:
            test_pattern_str = simpledialog.askstring("Manual Test Pattern",
                                                      f"Enter test pattern of size {input_size}: ")
            if test_pattern_str:
                test_pattern = [int(x) for x in test_pattern_str]
                if len(test_pattern) == input_size:
                    return test_pattern
                else:
                    messagebox.showerror("Error", f"Pattern length must be {input_size}.")
            else:
                return None
    elif choice == 'random':
        return random.choice(input_patterns)
    else:
        messagebox.showerror("Error", "Invalid test pattern method selected. Please choose 'manual' or 'random'.")
        return None


# Invoking the prgram functions:
if __name__ == "__main__":
    # Choose input method
    input_method = choose_input_method()

    if input_method == 'file':
        file_path = tk.filedialog.askopenfilename(title="Select input file", filetypes=[("Text files", "*.txt")])
        input_patterns = read_input_from_file(file_path)
    elif input_method == 'gui':
        input_size = int(simpledialog.askstring("Input Size", "Enter size of the input pattern: "))
        input_patterns = read_input_from_gui(input_size)
    else:
        print("Invalid input method selected. Please choose 'file' or 'gui'.")

    # Choose whether input vectors are unipolar or bipolar
    input_representation = simpledialog.askstring("Input Representation",
                                                  "Choose input representation ('unipolar' or 'bipolar'): ")
    if input_representation.lower().strip() == 'unipolar':
        unipolar = True
    elif input_representation.lower().strip() == 'bipolar':
        unipolar = False
    else:
        messagebox.showerror("Error", "Invalid input representation selected. Please choose 'unipolar' or 'bipolar'.")
        exit()

    # Initialize HAM
    ham = HopfieldAssociativeMemory(input_size=len(input_patterns[0]), unipolar=unipolar)

    # Train HAM
    ham.train(input_patterns)

    # Iterative testing
    while True:
        test_pattern = choose_test_pattern(input_patterns)
        if test_pattern:
            if isinstance(ham.recall(test_pattern), tuple):
                print("Output after convergence:", ham.recall(test_pattern)[0])
                print("Previous output:", ham.recall(test_pattern)[1])
            else:
                print("Network response after convergence:", ham.recall(test_pattern))
        else:
            break
