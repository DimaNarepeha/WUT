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
    def create_input_matrix_frame():
        nonlocal input_matrix_frame, input_entries
        input_matrix_frame = tk.Frame(gui)
        input_matrix_frame.grid(row=2, column=0, padx=10, pady=10)

        input_entries = []
        for i in range(input_size):
            row_entries = []
            for j in range(input_size):
                if representation == "unipolar":
                    entry_var = tk.StringVar()
                    entry = tk.Entry(input_matrix_frame, textvariable=entry_var, width=5)
                    entry.grid(row=i, column=j)
                    row_entries.append(entry_var)
                else:
                    entry_var = tk.StringVar(value="1")  # Initialize to empty string
                    entry = tk.Entry(input_matrix_frame, textvariable=entry_var, width=5)
                    entry.grid(row=i, column=j)
                    row_entries.append(entry_var)
            input_entries.append(row_entries)

        submit_button = tk.Button(input_matrix_frame, text="Submit", command=submit_input_matrix)
        submit_button.grid(row=input_size, columnspan=input_size, pady=10)

    def confirm_input_size():
        nonlocal input_matrix_frame
        if input_matrix_frame:
            input_matrix_frame.destroy()
        create_input_matrix_frame()

    def submit_input_matrix():
        nonlocal input_matrix
        input_matrix = []
        for i in range(input_size):
            row_values = []
            for j in range(input_size):
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
    input_matrix_frame = None
    input_entries = []

    gui = tk.Tk()
    gui.title("Input Matrix")

    confirm_size_button = tk.Button(gui, text="OK", command=confirm_input_size)
    confirm_size_button.grid(row=1, column=0, padx=10, pady=10)

    gui.mainloop()

    return input_matrix



def choose_representation():
    choice = simpledialog.askstring("Input Representation", "Choose input representation ('unipolar' or 'bipolar'): ")
    return choice.lower().strip()

def choose_input_method():
    choice = simpledialog.askstring("Input Method", "Choose input method ('gui' or 'file'): ")
    return choice.lower().strip()

def choose_test_pattern(input_size):
    input_rows = simpledialog.askinteger("Test Pattern", f"Enter the number of rows for the test pattern: ")
    if input_rows:
        input_vector = []
        for i in range(input_rows):
            row = simpledialog.askstring("Test Pattern", f"Enter row {i + 1} of the test pattern (separate values by space): ")
            if row:
                row = [int(x) if x.isdigit() else int(x) if x == '-1' else -1 for x in row.replace(",", "").split()]
                if len(row) == input_size:
                    input_vector.append(row)
                else:
                    messagebox.showerror("Error", f"Row length must be {input_size}.")
                    return None
            else:
                return None
        return input_vector
    return None


def continue_testing():
    choice = messagebox.askyesno("Continue Testing", "Do you want to continue testing?")
    return choice

if __name__ == "__main__":
    representation = choose_representation()
    input_method = choose_input_method()

    ham = HopfieldAssociativeMemory()

    if input_method == 'file':
        file_path = filedialog.askopenfilename(title="Select input file", filetypes=[("Text files", "*.txt")])
        input_patterns = read_input_from_file(file_path,representation)
        ham.train(input_patterns)
    elif input_method == 'gui':
        input_size = simpledialog.askinteger("Input Size", "Enter size of the input pattern: ")
        input_patterns = read_input_from_gui(input_size, representation)
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
