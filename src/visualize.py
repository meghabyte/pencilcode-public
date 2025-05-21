from collections import defaultdict
import urllib.parse
from string2string.distance import LevenshteinEditDistance
from string2string.misc import Tokenizer
import os
from matplotlib import pyplot as plt
import argparse
import pdb
import re

from src.utils import *

def plot_edit_distance(dists, fn, save_fig=True):
    x = [i / len(dists) for i in range(len(dists))]
    y = dists
    fig = plt.figure()
    plt.ylabel("Edit distance from final state")
    plt.xlabel("Index")
    plt.plot(x, y, color="magenta")
    if save_fig:
        plt.savefig(fn)
    else:
        plt.show()
    plt.close()


def write(file_name, data):
    # print(f"Writing to:\t{file_name}")
    with open(file_name, "w") as f:
        f.write(data)


def write_lines(file_name, data):
    print(f"Writing to:\t{file_name}")
    with open(file_name, "w") as f:
        f.writelines(data)


def render_image(html_file, image_file, time_budget=100000, do_return_html_file=False):
    from html2image import Html2Image

    flag = f"--virtual-time-budget={time_budget}"
    hti = Html2Image(
        custom_flags=[flag], 
        size=(500, 500),
        browser_executable='/usr/bin/google-chrome',
        )
    hti.browser.use_new_headless = None
    html = open(html_file, "r").read()
    # print("screenshot!")
    hti.screenshot(html_str=html, save_as="temp.png")
    # print("screenshot done!")
    # TODO: for some reason, doesn't work when image_file has directory (bc html2image creates temp files in a weird way)
    os.rename("temp.png", image_file)
    print(f"Wrote image to: {image_file}")
    if do_return_html_file:
        return html_file
    
def write_html_file(program_code, html_file, language="coffeescript"):
    with open(f"data/{language}_template.html", "r") as f:
        data = f.read()
        data = data.replace("INSERT_CODE_HERE", program_code)
    # print(f'Writing html to {html_file}')
    write(html_file, data)
    
def visualize_program(program_code, image_file="image.png", time_budget=100000, html_file='temp.html', do_return_html_file=False):
    write_html_file(program_code, html_file) 

    html_file = render_image(html_file, image_file, time_budget=time_budget, do_return_html_file=do_return_html_file)
    if do_return_html_file:
        return html_file


# TODO: this is old code from when we were first working with sample data
def run_visualize_program(
    solutions_dict=None,
    username="css6room2",
    task="Moon",
    save_fig=True,
    render_images=False,
):
    directory = "visualizations/" + task.replace("/", "-") + "/" + username
    if not os.path.exists(directory):
        os.makedirs(directory)

    if solutions_dict is None:
        solutions_dict = get_solutions_dict()

    edit_dist = LevenshteinEditDistance(
        insert_weight=0.0
    )  # override default insertion weight of 1.0

    program_states = solutions_dict[username][task]
    parsed_states = []
    dists, word_dists = [], []
    count = 0
    final_state = parse_state(program_states[-1])
    for state_idx, state in enumerate(program_states):
        ps = parse_state(state)
        parsed_states.append(ps)
        dists.append(edit_dist.compute(ps, final_state))
        ps_tokens = re.split("[\n, ]+", ps)
        final_state_tokens = re.split("[\n, ]+", final_state)
        word_dist = edit_dist.compute(ps_tokens, final_state_tokens)
        word_dists.append(word_dist)
        with open("./template.html", "r") as f:
            data = f.read()
            print(ps)
            data = data.replace("INSERT_CODE_HERE", ps)
        html_file = os.path.join(directory, f"state_{count}.html")
        write(html_file, data)
        if render_images:
            image_dir = os.path.join(directory, "images")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            image_file = os.path.join(image_dir, f"state_{count}.png")
            render_image(html_file, image_file)
        count += 1
    write_lines(directory + "/parsed_program_states", parsed_states)
    plot_edit_distance(dists, directory + "/dists", save_fig=save_fig)
    plot_edit_distance(word_dists, directory + "/word_dists", save_fig=save_fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", default="css6room2")
    parser.add_argument("-t", "--task", default="moon")

    args = parser.parse_args()

    solutions_dict = get_solutions_dict()
    if args.username == "ALL_STUDENTS":
        for student_key in solutions_dict.keys():
            if args.task in solutions_dict[student_key]:
                run_visualize_program(
                    username=student_key, task=args.task, solutions_dict=solutions_dict
                )
    else:
        run_visualize_program(
            username=args.username,
            task=args.task,
            solutions_dict=solutions_dict,
            render_images=True,
        )
