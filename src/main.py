from src.image_generator import generate_multiple_images


def main() -> None:
    """Main function to generate and save fiber images with annotations."""

    generate_multiple_images(num_images=1,
                             start_index=0,
                             image_size=(2472, 3296),
                             save=True, save_format='yolo',
                             image_list_file='test.txt',
                             save_dir='../data/images/',
                             label_dir='../data/images/',
                             gui=True)


if __name__ == '__main__':
    main()
