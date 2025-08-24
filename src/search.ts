const FlexSearch = require('flexsearch');

export class SearchManager {
    private keywordIndex: any;

    constructor() {
        this.keywordIndex = new FlexSearch.Index({
            tokenize: "forward"
        });
    }

    add(id: string, text: string) {
        this.keywordIndex.add(id, text);
    }

    search(query: string) {
        return this.keywordIndex.search(query);
    }

    remove(id: string) {
        this.keywordIndex.remove(id);
    }

    clear() {
        this.keywordIndex = new FlexSearch.Index({
            tokenize: "forward"
        });
    }
}
