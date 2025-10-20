/^impl<U: Into<IValue>> Extend<U> for IArray {/,/^}$/ {
    s/^/\/\/ /
}
/^impl<U: Into<IValue>> FromIterator<U> for IArray {/,/^}$/ {
    s/^/\/\/ /
}
/^impl<T: Into<IValue>> From<Vec<T>> for IArray {/,/^}$/ {
    s/^/\/\/ /
}
/^impl<T: Into<IValue> \+ Clone> From<&\[T\]> for IArray {/,/^}$/ {
    s/^/\/\/ /
}
