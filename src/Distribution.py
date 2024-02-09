import os
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt


def retrieve_file_subdir(dir):
    data = {}
    for foldername, subdirectorys, filenames in os.walk(dir):
        base_foldername = foldername
        file_names = []
        if subdirectorys:
            for subdirectory in subdirectorys:
                for filename in filenames:
                    file_names.append(foldername + "/" + subdirectory + "/" + filename)
        else :
            for filename in filenames:
                    file_names.append(foldername + "/" + filename)
        data[base_foldername] = pd.Series(file_names)
    df = pd.concat(data, axis=1)  # To deal with dict
    df.dropna(axis=1, how="all", inplace=True)
    return df


def plot_value(df):
    file_count = df.count()  # en trouve 1 de moins
    plt.figure(figsize=(12, 6))

    # Plot bar chart
    plt.subplot(1, 2, 1)
    plt.bar(df.columns, file_count, color='lightcoral')
    plt.xticks(rotation=90)
    plt.title('File Count per Directory (Bar Chart)')

    # Plot pie chart
    plt.subplot(1, 2, 2)
    plt.pie(file_count, labels=df.columns, autopct='%1.1f%%', startangle=90,
            colors=['lightgreen', 'lightcoral', 'skyblue'])
    plt.title('File Count Distribution (Pie Chart)')

    plt.tight_layout()
    plt.show()


def main():
    try:
        if not os.path.isdir(argv[1]):
            print("Please enter a directory as a parametter")
            return 1
        df = retrieve_file_subdir(argv[1])
        plot_value(df)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()
