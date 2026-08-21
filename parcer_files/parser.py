import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Query, QueryCursor
import os


class GoAstParser:
    def __init__(self):
        self.language = Language(tsgo.language())
        self.parser = Parser(self.language)

        # Паттерн остается тем же
        self.call_query = Query(self.language, """
            (call_expression (identifier) @call_name)
            (call_expression (selector_expression) @call_name)
        """)

        self.cursor = QueryCursor(self.call_query)

    def parse_file(self, file_path: str) -> list:
        if not os.path.exists(file_path):
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        all_calls = []
        try:
            # ИСПОЛЬЗУЕМ matches вместо captures!
            # Согласно вашему stub: returns List[Tuple[int, Dict[str, Node]]]
            # Мы передаем node и query в matches
            matches = self.cursor.matches(root_node, self.call_query)

            for pattern_index, captures_dict in matches:
                # captures_dict это { 'call_name': node }
                for capture_name, node in captures_dict.items():
                    start_byte = node.start_byte
                    end_byte = node.end_byte
                    all_calls.append((start_byte, end_byte, code[start_byte:end_byte]))

        except Exception as e:
            print(f"Error during query matches: {e}")

        chunks = []
        self._traverse_tree(root_node, code, file_path, chunks, all_calls)
        return chunks

    def _traverse_int(self, node, source_code, file_path, chunks, all_calls):
        # Вспомогательный метод для обхода (рекурсия)
        pass

    def _traverse_tree(self, node, source_code, file_path, chunks, all_calls):
        check_types = ['function_declaration', 'method_declaration', 'type_declaration']

        if node.type in check_types:
            start_byte = node.start_byte
            end_byte = node.end_byte
            chunk_content = source_code[start_byte:end_byte]

            metadata = self._extract_base_metadata(node, source_code, file_path)

            calls = set()
            for call_start, call_end, call_name in all_calls:
                # Проверяем, находится ли вызов внутри текущего узла
                if node.start_byte <= call_start < node.end_byte:
                    calls.add(call_name)

            metadata['calls'] = list(calls)

            chunks.append({
                "content": chunk_content,
                "metadata": metadata
            })

        for child in node.children:
            self._traverse_tree(child, source_code, file_path, chunks, all_calls)

    def _extract_base_metadata(self, node, source_code, file_path):
        metadata = {
            "file_path": file_path,
            "name": "unknown",
            "type": node.type,
            "start_line": node.start_point[0] + 1
        }
        for child in node.children:
            if child.type == 'identifier':
                metadata["name"] = source_code[child.start_byte:child.end_byte]
                break
        return metadata


class DependencyGraphBuilder:
    def __init__(self):
        self.graph = {}

    def build_from_chunks(self, chunks):
        for chunk in chunks:
            caller = chunk['metadata'].get('name')
            callees = chunk['metadata'].get('calls', [])

            if caller and caller != 'unknown':
                if caller not in self.graph:
                    self.graph[caller] = []
                self.graph[caller].extend(callees)

        for caller in self.graph:
            self.graph[caller] = list(set(self.graph[caller]))


if __name__ == "__main__":
    test_code = """
package main
import "fmt"
func Connect() { fmt.Println("Connect") }
func Query() { Connect(); fmt.Println("Query") }
"""
    test_file = "test.go"
    with open(test_file, "w") as f:
        f.write(test_code)

    parser = GoAstParser()
    chunks = parser.parse_file(test_file)

    builder = DependencyGraphBuilder()
    builder.build_from_chunks(chunks)

    print("--- Results ---")
    for caller, callees in sorted(builder.graph.items()):
        print(f"{caller} -> {callees}")

    if os.path.exists(test_file): os.remove(test_file)
