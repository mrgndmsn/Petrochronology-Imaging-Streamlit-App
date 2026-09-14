"""Keep input widget values through navigation, without retaining button presses."""
import ast
from functools import lru_cache
from pathlib import Path
import streamlit as st

INPUTS={'selectbox','multiselect','radio','segmented_control','checkbox','toggle',
        'number_input','slider','select_slider','text_input','text_area','color_picker','date_input','time_input'}


def preserve_widget_state():
    # Reassigning a keyed widget value detaches it from Streamlit's cleanup for
    # this run. Do this before rendering any widgets, including navigation.
    for key in st.session_state.get('_remembered_widget_keys', []):
        if key in st.session_state:
            st.session_state[key]=st.session_state[key]


def remembered_input(identity, method, *args, **kwargs):
    key=kwargs.setdefault('key','input::'+identity)
    registry=st.session_state.setdefault('_remembered_widget_keys', [])
    if key not in registry: registry.append(key)
    kind=method.__name__
    if kind in {'selectbox','multiselect','radio','segmented_control','select_slider'}:
        options=kwargs.get('options', args[1] if len(args)>1 else [])
        options=list(options)
        old=st.session_state.get(key)
        if old is not None:
            if isinstance(old,list):
                valid=[v for v in old if v in options]
                if old and not valid and options: valid=options
                if valid != old: st.session_state[key]=valid
            elif old not in options:
                del st.session_state[key]
    return method(*args,**kwargs)


@lru_cache(maxsize=64)
def compiled_page(path,mtime):
    tree=ast.parse(Path(path).read_text(),filename=path)
    class Inputs(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr in INPUTS:
                node=ast.copy_location(ast.Call(func=ast.Name(id='_remembered_input',ctx=ast.Load()),
                    args=[ast.Constant(f'{path}:{node.lineno}:{node.col_offset}'),node.func]+node.args,
                    keywords=node.keywords),node)
            return node
    return compile(ast.fix_missing_locations(Inputs().visit(tree)),path,'exec')


def run_page(path):
    path=str(path)
    calls={}
    def page_input(identity,method,*args,**kwargs):
        # A comprehension can create several controls at the same source line.
        # Include the displayed label and occurrence, not only its line number.
        if 'key' not in kwargs:
            label=str(kwargs.get('label',args[0] if args else ''))
            base=identity+'::'+label
            count=calls.get(base,0);calls[base]=count+1
            identity=base+'::'+str(count)
        return remembered_input(identity,method,*args,**kwargs)
    exec(compiled_page(path,Path(path).stat().st_mtime_ns),
         {'__name__':'__main__','__file__':path,'_remembered_input':page_input})
