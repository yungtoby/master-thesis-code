from yahpo_gym import BenchmarkSet


def main():
    b = BenchmarkSet("lcbench")
    instances = [str(x) for x in b.instances]

    print(f"Found {len(instances)} LCBench instances:")
    for i, inst in enumerate(instances):
        print(f"{i}: {inst}")


if __name__ == "__main__":
    main()