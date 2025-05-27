from roof_api_yolo_word import detect_roof_type  # Update with actual function name

def main():
    while True:
        path = input("Enter roof image filename (or 'done' to exit): ")
        if path.lower() == 'done':
            break

        try:
            result = detect_roof_type(path)
            print(f"🧠 Detected Roof Type: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
