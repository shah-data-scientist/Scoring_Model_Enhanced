import os
import sys
from sqlalchemy_schemadisplay import create_schema_graph

# Add the project root to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models import Base
from backend.database import engine

def generate_schema_dot():
    # Create the graph
    graph = create_schema_graph(
        metadata=Base.metadata,
        engine=engine,
        show_datatypes=True,
        show_indexes=True,
        rankdir='LR',
        concentrate=False
    )
    
    # Write to dot file manually to avoid pydot's dependency on Graphviz binary
    output_path = os.path.join(os.path.dirname(__file__), '../backend/schema.dot')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_string())
        
    print(f"Schema DOT file generated at: {output_path}")

if __name__ == "__main__":
    generate_schema_dot()
