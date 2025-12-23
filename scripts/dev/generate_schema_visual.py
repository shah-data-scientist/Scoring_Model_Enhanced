import os
import sys
import inspect
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.sql import sqltypes

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models import Base
# Import all models to ensure they are registered
from backend import models

def get_mermaid_type(python_type):
    """Map SQLAlchemy types to generic names for display."""
    type_map = {
        sqltypes.Integer: "INTEGER",
        sqltypes.String: "STRING",
        sqltypes.Boolean: "BOOLEAN",
        sqltypes.DateTime: "DATETIME",
        sqltypes.Float: "FLOAT",
        sqltypes.JSON: "JSON",
        sqltypes.Text: "TEXT",
        sqltypes.BigInteger: "BIGINT",
        sqltypes.Enum: "ENUM"
    }
    
    for sqla_type, name in type_map.items():
        if isinstance(python_type, sqla_type) or (isinstance(python_type, type) and issubclass(python_type, sqla_type)):
            return name
            
    return str(python_type).split('(')[0]

def generate_mermaid_er():
    mermaid_lines = ["erDiagram"]
    
    # Get all mappers
    from sqlalchemy.orm import class_mapper
    
    # Find all subclasses of Base
    models_list = [
        cls for cls in Base.__subclasses__()
        if hasattr(cls, '__tablename__')
    ]
    
    # Generate Entities
    for model in models_list:
        table_name = model.__tablename__
        mermaid_lines.append(f"    {table_name} {{")
        
        mapper = class_mapper(model)
        for column in mapper.columns:
            col_type = get_mermaid_type(column.type)
            # Add PK/FK indicators
            constraints = []
            if column.primary_key:
                constraints.append("PK")
            if column.foreign_keys:
                constraints.append("FK")
            
            constraint_str = ", ".join(constraints)
            if constraint_str:
                constraint_str = f" {constraint_str}"
                
            mermaid_lines.append(f"        {col_type} {column.name}{constraint_str}")
            
        mermaid_lines.append("    }")

    # Generate Relationships
    # This is a bit heuristic since SQLAlchemy relationships are object-level
    processed_rels = set()
    
    for model in models_list:
        mapper = class_mapper(model)
        source_table = model.__tablename__
        
        for prop in mapper.iterate_properties:
            if isinstance(prop, RelationshipProperty):
                target_model = prop.mapper.class_
                target_table = target_model.__tablename__
                
                # Check direction (basic heuristic)
                if prop.direction.name == 'MANYTOONE':
                    rel_type = "}o--||" # Many to One
                elif prop.direction.name == 'ONETOMANY':
                    rel_type = "||--o{" # One to Many
                elif prop.direction.name == 'MANYTOMANY':
                    rel_type = "}o--o{"
                else:
                    rel_type = "|o--o|"
                
                # Create a unique key for the relationship to avoid duplicates
                # Sort tables to handle bidirectional relationships gracefully
                t1, t2 = sorted([source_table, target_table])
                rel_key = f"{t1}-{t2}"
                
                # Only add if we haven't seen this pair, or if it's a specific directional one we want to capture
                # For ER diagrams, usually we just want the line.
                # Mermaid handles duplicates okay, but let's be clean.
                # Actually, capturing from the "Many" side (ForeignKey side) is usually enough for the structure.
                
                # Let's rely on ForeignKeys for the lines to be accurate to the schema
                pass

    # Alternative Relationship Generation using Foreign Keys (more accurate for ER)
    for model in models_list:
        table_name = model.__tablename__
        mapper = class_mapper(model)
        
        for column in mapper.columns:
            for fk in column.foreign_keys:
                target_table = fk.column.table.name
                # Relationship: Source (FK side) }o--|| Target (PK side)
                line = f"    {table_name} }}o--|| {target_table} : \"{column.name}\""
                mermaid_lines.append(line)

    return "\n".join(mermaid_lines)

def create_html_file(mermaid_content):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Schema - Scoring Model</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Database Schema</h1>
        <p>Generated automatically from SQLAlchemy models.</p>
        <div class="mermaid">
{mermaid_content}
        </div>
    </div>
</body>
</html>
"""
    return html_content

if __name__ == "__main__":
    try:
        mermaid_code = generate_mermaid_er()
        html = create_html_file(mermaid_code)
        
        output_path = os.path.join(os.path.dirname(__file__), '../backend/schema_visual.html')
        output_path = os.path.abspath(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Successfully generated visual schema at: {output_path}")
    except Exception as e:
        print(f"Error generating schema: {e}")
        import traceback
        traceback.print_exc()
