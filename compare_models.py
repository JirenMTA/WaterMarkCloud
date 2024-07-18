import re
import matplotlib.pyplot as plt
import test_model1.test_model1 as md1
import test_model2.test_model2 as md2
import test_model3.test_model3 as md3
import test_model4.test_model4 as md4


def work_on_1_input():
    md1.test_and_show_1_input()
    md2.test_and_show_1_input()
    md3.test_and_show_1_input()
    md4.test_and_show_1_input(arch='FPN')
    md4.test_and_show_1_input(arch='UNet')

def log_losses():
    md1.write_losses_of_model()
    md2.write_losses_of_model()
    md3.write_losses_of_model()
    md4.write_losses_of_model(arch='FPN')
    md4.write_losses_of_model(arch='UNet')

def show_graphic():
    with open("./result_loss.txt", 'r') as f:
        content = f.read()

    model_pattern = re.compile(r'\*{20} (.*?) \*{20}')
    loss_pattern = re.compile(r'(\d+\.\d+) - Num inputs:')
    input_pattern = re.compile(r'Num inputs:\s(\d+)')

    model_sections = model_pattern.split(content)

    model_losses = {}
    for i in range(1, len(model_sections), 2):
        model_name = model_sections[i].strip()
        section_content = model_sections[i + 1]
        losses = loss_pattern.findall(section_content)
        inputs = input_pattern.findall(section_content)

        float_losses = list(map(float, losses))
        int_inputs = list(map(int, inputs))

        model_losses[model_name] = [int_inputs, float_losses]

    plt.figure(figsize=(10, 6))
    for model_name, values in model_losses.items():
        inputs = values[0]
        losses = values[1]

        plt.grid(True)
        plt.plot(inputs, losses, marker='o', label=model_name)

    plt.xlabel('Number of Inputs')
    plt.ylabel('Losses')
    plt.title('Model Losses vs Number of Inputs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    work_on_1_input()
    #log_losses()
    #show_graphic()