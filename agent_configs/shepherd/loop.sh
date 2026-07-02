{
    cat prompt.md
    printf '\n\nRequirement: Start by reading the shepherd skill.\n'
} | claude --permission-mode bypassPermissions -p
