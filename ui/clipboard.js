
function copyToClipboard(text) {
    if (text) {
        navigator.clipboard.writeText(text);
        return `Tags copied: ${text.substring(0, 50)}...`;
    }
    return "No tags to copy";
}
