## Key points
- We want to have a modern feel cli app.
- Always use the display.rs utils functions to display to the terminal to keep a uniform theme across the app.
- If a display functionality is needed and not in display.rs, add it and use rich_rust to display it. Ensure it follows the theme.
- The `parameters` module contains the generic parameters options such as Leiden, HNSW. They also contain a function to 
return a BTree<string, string> containing the key and value of each field in the struct.
- The `kernel_parameters` module contains the generic parameters options related to kernels.
- Helper functions are implemented in `utils`.

## Rich rust documentation
Rich rust documentation is in `llm_docs/rich_rust` in the form of markdown files.