import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog

class HopfieldAssociativeMemory:
    def __init__(self):
        self.weights = None

    def train(self, input_patterns):
        num_neurons = len(input_patterns[0])
        self.weights = np.zeros((num_neurons, num_neurons))
        for pattern in input_patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern)
        current_pattern = input_pattern.copy()
        prev_pattern = np.zeros_like(current_pattern)
        cycle_detected = False

        outputs = []

        for i in range(max_iterations):
            next_pattern = np.sign(np.dot(self.weights, current_pattern))

            if np.array_equal(next_pattern, current_pattern):
                return outputs, self.calculate_energy(current_pattern)

            if np.array_equal(next_pattern, prev_pattern):
                cycle_detected = True
                break

            outputs.append(current_pattern.tolist())

            prev_pattern = current_pattern.copy()
            current_pattern = next_pattern

        if cycle_detected:
            return outputs + [prev_pattern.tolist()], "Cycle of length two detected."
        else:
            return "Network did not converge.", None

    def calculate_energy(self, pattern):
        return -0.5 * np.dot(np.dot(pattern, self.weights), pattern)

def read_input_from_file(file_path, representation):
    valid_characters = {'unipolar': {'0', '1'}, 'bipolar': {'1', '-1'}}
    with open(file_path, 'r') as file:
        input_patterns = []
        expected_length = None
        for line in file:
            pattern = []
            for part in line.strip().split('-'):
                for char in part:
                    if char not in valid_characters[representation]:
                        messagebox.showerror("Error", f"Invalid character '{char}' found in input file.")
                        return []
                    pattern.append(int(char))
            if expected_length is None:
                expected_length = len(pattern)
            elif len(pattern) != expected_length:
                messagebox.showerror("Error", "Input patterns must have the same length.")
                return []
            input_patterns.append(pattern)
    return input_patterns

def read_input_from_gui(num_columns, num_rows, representation):
    def submit_input_matrix():
        nonlocal input_matrix
        input_matrix = []
        for i in range(num_rows):
            row_values = []
            for j in range(num_columns):
                value = input_entries[i][j].get()
                if representation == "unipolar":
                    if value not in ["0", "1"]:
                        messagebox.showerror("Error", "Values must be 0 or 1.")
                        return
                    row_values.append(int(value))
                else:
                    if value not in ["-1", "1"]:
                        messagebox.showerror("Error", "Values must be -1 or 1.")
                        return
                    row_values.append(int(value))
            input_matrix.append(row_values)
        gui.destroy()

    input_matrix = None

    gui = tk.Tk()
    gui.title("Input Matrix")

    input_entries = []
    for i in range(num_rows):
        row_entries = []
        for j in range(num_columns):
            entry_var = tk.StringVar()
            entry = tk.Entry(gui, textvariable=entry_var, width=5)
            entry.grid(row=i, column=j)
            row_entries.append(entry_var)
        input_entries.append(row_entries)

    submit_button = tk.Button(gui, text="Submit", command=submit_input_matrix)
    submit_button.grid(row=num_rows, columnspan=num_columns, pady=10)

    gui.mainloop()

    return input_matrix

def choose_representation():
    def submit_choice():
        nonlocal representation_choice
        representation_choice = var.get()
        gui.destroy()

    representation_choice = None
    gui = tk.Tk()
    gui.title("Select Representation")

    var = tk.StringVar(value="unipolar")

    tk.Label(gui, text="Choose input representation:").pack(pady=10)
    tk.Radiobutton(gui, text="Unipolar", variable=var, value="unipolar").pack(anchor=tk.W)
    tk.Radiobutton(gui, text="Bipolar", variable=var, value="bipolar").pack(anchor=tk.W)
    tk.Button(gui, text="Submit", command=submit_choice).pack(pady=10)

    gui.mainloop()

    return representation_choice

def choose_input_method():
    def submit_choice():
        nonlocal input_method_choice
        input_method_choice = var.get()
        gui.destroy()

    input_method_choice = None
    gui = tk.Tk()
    gui.title("Select Input Method")

    var = tk.StringVar(value="gui")

    tk.Label(gui, text="Choose input method:").pack(pady=10)
    tk.Radiobutton(gui, text="GUI", variable=var, value="gui").pack(anchor=tk.W)
    tk.Radiobutton(gui, text="File", variable=var, value="file").pack(anchor=tk.W)
    tk.Button(gui, text="Submit", command=submit_choice).pack(pady=10)

    gui.mainloop()

    return input_method_choice

def get_vector_and_matrix_size():
    def submit_choice():
        nonlocal num_columns, num_rows
        try:
            num_columns = int(entry_columns.get())
            num_rows = int(entry_rows.get())
            gui.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values.")

    num_columns, num_rows = None, None

    gui = tk.Tk()
    gui.title("Matrix Size Input")

    tk.Label(gui, text="Enter length of the vector (number of columns):").grid(row=0, column=0, padx=10, pady=10)
    entry_columns = tk.Entry(gui)
    entry_columns.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(gui, text="Enter number of rows:").grid(row=1, column=0, padx=10, pady=10)
    entry_rows = tk.Entry(gui)
    entry_rows.grid(row=1, column=1, padx=10, pady=10)

    submit_button = tk.Button(gui, text="Submit", command=submit_choice)
    submit_button.grid(row=2, columnspan=2, pady=10)

    gui.mainloop()

    return num_columns, num_rows

def choose_test_pattern(input_size):
    input_vector = simpledialog.askstring("Test Pattern", f"Enter test pattern of size {input_size}: ")
    if input_vector:
        input_vector = [int(x) if x.isdigit() else int(x) if x == '-1' else -1 for x in input_vector.replace(",", "").split()]
        if len(input_vector) == input_size:
            return input_vector
        else:
            messagebox.showerror("Error", f"Pattern length must be {input_size}.")
    return None

def continue_testing():
    def submit_choice():
        nonlocal continue_choice
        continue_choice = var.get() == "yes"
        gui.destroy()

    continue_choice = None
    gui = tk.Tk()
    gui.title("Continue Testing")

    var = tk.StringVar(value="yes")

    tk.Label(gui, text="Do you want to continue testing?").pack(pady=10)
    tk.Radiobutton(gui, text="Yes", variable=var, value="yes").pack(anchor=tk.W)
    tk.Radiobutton(gui, text="No", variable=var, value="no").pack(anchor=tk.W)
    tk.Button(gui, text="Submit", command=submit_choice).pack(pady=10)

    gui.mainloop()

    return continue_choice

if __name__ == "__main__":
    representation = choose_representation()
    input_method = choose_input_method()

    ham = HopfieldAssociativeMemory()

    if input_method == 'file':
        file_path = filedialog.askopenfilename(title="Select input file", filetypes=[("Text files", "*.txt")])
        input_patterns = read_input_from_file(file_path, representation)
        if input_patterns:
            ham.train(input_patterns)
    elif input_method == 'gui':
        num_columns, num_rows = get_vector_and_matrix_size()
        input_patterns = read_input_from_gui(num_columns, num_rows, representation)
        if input_patterns:
            ham.train(input_patterns)
    else:
        messagebox.showerror("Error", "Invalid input method selected. Please choose 'gui' or 'file'.")

    while True:
        test_pattern = choose_test_pattern(len(input_patterns[0]))
        if test_pattern:
            output, energy = ham.recall(test_pattern)
            messagebox.showinfo("Output", f"Network Response: {output}\nEnergy: {energy}")
        else:
            break

        if not continue_testing():
            break
