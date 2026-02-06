from epinet.baseline import infer_netrate_baseline

def main():
    # We expect this to fail because files don't exist,
    # but it should get past argparse without "unrecognized arguments".
    infer_netrate_baseline("dummy_cascades.csv", "dummy_out.csv")

if __name__ == "__main__":
    main()
