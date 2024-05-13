import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk


class HAM:
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
        input_patterns = np.array(input_pattern)
        outputs = []
        energies = []

        for input_pattern in input_patterns:
            input_pattern = np.array(input_pattern)
            current_pattern = input_pattern.copy()
            prev_pattern = np.zeros_like(current_pattern)
            cycle_detected = False

            for i in range(max_iterations):
                next_pattern = np.sign(np.dot(self.weights, current_pattern))

                if np.array_equal(next_pattern, current_pattern):
                    outputs.append(current_pattern.tolist())
                    energies.append(self.calculate_energy(current_pattern))
                    break

                if np.array_equal(next_pattern, prev_pattern):
                    cycle_detected = True
                    break

                prev_pattern = current_pattern.copy()
                current_pattern = next_pattern

            if cycle_detected:
                energy = self.calculate_energy(prev_pattern)
                outputs.append(prev_pattern.tolist())
                energies.append(energy)
                outputs.append("Cycle of length two detected.")
                energies.append(None)

        if not outputs:
            return "Network did not converge.", None
        else:
            return outputs, energies

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


def read_input_from_gui(input_size, representation):
    def create_matrix_frame():
        nonlocal matrix_frame, matrix_entries
        matrix_frame = ttk.Frame(gui)
        matrix_frame.grid(row=2, column=0, padx=10, pady=10)

        matrix_entries = []
        for i in range(input_size):
            row_entries = []
            for j in range(input_size):
                entry_var = tk.StringVar(value="1" if representation == "bipolar" else "")
                entry = ttk.Entry(matrix_frame, textvariable=entry_var, width=5)
                entry.grid(row=i, column=j)
                row_entries.append(entry_var)
            matrix_entries.append(row_entries)

        submit_btn = ttk.Button(matrix_frame, text="Submit Pattern", command=submit_matrix)
        submit_btn.grid(row=input_size, columnspan=input_size, pady=10)

    def confirm_size():
        nonlocal matrix_frame
        if matrix_frame:
            matrix_frame.destroy()
        create_matrix_frame()

    def submit_matrix():
        nonlocal matrix
        matrix = []
        for i in range(input_size):
            row_values = []
            for j in range(input_size):
                value = matrix_entries[i][j].get()
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
            matrix.append(row_values)
        gui.destroy()

    matrix = None
    matrix_frame = None
    matrix_entries = []

    gui = tk.Tk()
    gui.title("Matrix Input")

    main_frame = ttk.Frame(gui, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the size of the input pattern:")
    size_label.grid(row=0, column=0, padx=5, pady=5)

    size_btn = ttk.Button(main_frame, text="Set Size", command=confirm_size)
    size_btn.grid(row=1, column=0, padx=5, pady=5)

    gui.mainloop()

    return matrix


def choose_representation():
    global representation_choice
    gui = tk.Tk()
    gui.title("Select Representation")

    representation = tk.StringVar(value="unipolar")

    def submit_representation():
        global representation_choice
        representation_choice = representation.get()
        gui.destroy()

    ttk.Label(gui, text="Choose input representation:").grid(column=0, row=0, padx=10, pady=10)

    ttk.Radiobutton(gui, text="Unipolar", variable=representation, value="unipolar").grid(column=0, row=1, padx=10,
                                                                                          pady=5)
    ttk.Radiobutton(gui, text="Bipolar", variable=representation, value="bipolar").grid(column=0, row=2, padx=10,
                                                                                        pady=5)

    ttk.Button(gui, text="Submit", command=submit_representation).grid(column=0, row=3, padx=10, pady=10)

    gui.mainloop()
    return representation_choice


def choose_input_method():
    global input_method_choice
    gui = tk.Tk()
    gui.title("Select Input Method")

    method = tk.StringVar(value="gui")

    def submit_method():
        global input_method_choice
        input_method_choice = method.get()
        gui.destroy()

    ttk.Label(gui, text="Choose input method:").grid(column=0, row=0, padx=10, pady=10)

    ttk.Radiobutton(gui, text="GUI", variable=method, value="gui").grid(column=0, row=1, padx=10, pady=5)
    ttk.Radiobutton(gui, text="File", variable=method, value="file").grid(column=0, row=2, padx=10, pady=5)

    ttk.Button(gui, text="Submit", command=submit_method).grid(column=0, row=3, padx=10, pady=10)

    gui.mainloop()
    return input_method_choice


def choose_test_pattern(input_size):
    input_rows = simpledialog.askinteger("Test Pattern Entry", "How many rows for the test pattern?")
    if input_rows:
        input_vector = []
        for i in range(input_rows):
            row = simpledialog.askstring("Pattern Row Entry", f"Enter values for row {i + 1} (space-separated): ")
            if row:
                row = [int(x) if x.isdigit() else int(x) if x == '-1' else -1 for x in row.replace(",", "").split()]
                if len(row) == input_size:
                    input_vector.append(row)
                else:
                    messagebox.showerror("Error", f"Each row must be {input_size} values long.")
                    return None
            else:
                return None
        return input_vector
    return None


def continue_testing():
    global continue_choice
    gui = tk.Tk()
    gui.title("Continue Testing?")

    continue_var = tk.StringVar(value="yes")

    def submit_continue():
        global continue_choice
        continue_choice = (continue_var.get() == "yes")
        gui.destroy()

    ttk.Label(gui, text="Do you want to continue testing?").grid(column=0, row=0, padx=10, pady=10)

    ttk.Radiobutton(gui, text="Yes", variable=continue_var, value="yes").grid(column=0, row=1, padx=10, pady=5)
    ttk.Radiobutton(gui, text="No", variable=continue_var, value="no").grid(column=0, row=2, padx=10, pady=5)

    ttk.Button(gui, text="Submit", command=submit_continue).grid(column=0, row=3, padx=10, pady=10)

    gui.mainloop()
    return continue_choice


if __name__ == "__main__":
    representation = choose_representation()
    input_method = choose_input_method()

    ham = HAM()

    if input_method == 'file':
        file_path = filedialog.askopenfilename(title="Choose input file", filetypes=[("Text files", "*.txt")])
        input_patterns = read_input_from_file(file_path, representation)
        ham.train(input_patterns)
    elif input_method == 'gui':
        input_size = simpledialog.askinteger("Pattern Size", "Enter the size of the input pattern: ")
        input_patterns = read_input_from_gui(input_size, representation)
        ham.train(input_patterns)
    else:
        messagebox.showerror("Error", "Invalid input method. Please choose 'gui' or 'file'.")

    while True:
        test_pattern = choose_test_pattern(len(input_patterns[0]))
        if test_pattern:
            output, energy = ham.recall(test_pattern)
            messagebox.showinfo("Network Output", f"Response: {output}\nEnergy: {energy}")
        else:
            break

        if not continue_testing():
            break
