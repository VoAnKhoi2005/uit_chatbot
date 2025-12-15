// IndexedDB for storing conversations, messages, and source documents

const DB_NAME = "UIT_Chatbot_DB";
const DB_VERSION = 1;

export interface Conversation {
  id?: number;
  title: string;
  createdAt: number;
  updatedAt: number;
}

export interface ChatMessage {
  id?: number;
  conversationId: number;
  role: "user" | "bot";
  text: string;
  timestamp: string;
  sources?: SourceDocument[];
  questionType?: string;
  createdAt: number;
}

export interface SourceDocument {
  id?: number;
  messageId: number;
  conversationId: number;
  article_id?: string;
  clause_id?: string;
  text: string;
  doc_id?: string;
  title?: string;
  doc_title?: string;
  so_hieu?: string;
  createdAt: number;
}

class ChatDatabase {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Conversations store
        if (!db.objectStoreNames.contains("conversations")) {
          const conversationStore = db.createObjectStore("conversations", {
            keyPath: "id",
            autoIncrement: true,
          });
          conversationStore.createIndex("createdAt", "createdAt", { unique: false });
          conversationStore.createIndex("updatedAt", "updatedAt", { unique: false });
        }

        // Messages store
        if (!db.objectStoreNames.contains("messages")) {
          const messageStore = db.createObjectStore("messages", {
            keyPath: "id",
            autoIncrement: true,
          });
          messageStore.createIndex("conversationId", "conversationId", { unique: false });
          messageStore.createIndex("createdAt", "createdAt", { unique: false });
        }

        // Source documents store
        if (!db.objectStoreNames.contains("sources")) {
          const sourceStore = db.createObjectStore("sources", {
            keyPath: "id",
            autoIncrement: true,
          });
          sourceStore.createIndex("messageId", "messageId", { unique: false });
          sourceStore.createIndex("conversationId", "conversationId", { unique: false });
        }
      };
    });
  }

  // Conversations
  async createConversation(title: string): Promise<number> {
    const now = Date.now();
    const conversation: Conversation = {
      title,
      createdAt: now,
      updatedAt: now,
    };
    return this.add("conversations", conversation);
  }

  async getConversation(id: number): Promise<Conversation | null> {
    return this.get("conversations", id);
  }

  async getAllConversations(): Promise<Conversation[]> {
    const conversations = await this.getAll("conversations");
    return conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async updateConversation(id: number, updates: Partial<Conversation>): Promise<void> {
    const conversation = await this.getConversation(id);
    if (!conversation) throw new Error("Conversation not found");
    const updated = { ...conversation, ...updates, updatedAt: Date.now() };
    await this.put("conversations", updated);
  }

  async deleteConversation(id: number): Promise<void> {
    await this.delete("conversations", id);
    await this.deleteMessagesByConversation(id);
    await this.deleteSourcesByConversation(id);
  }

  // Messages
  async addMessage(message: Omit<ChatMessage, "id" | "createdAt">): Promise<number> {
    const msg: ChatMessage = {
      ...message,
      createdAt: Date.now(),
    };
    const messageId = await this.add("messages", msg);

    // Save sources separately
    if (message.sources && message.sources.length > 0) {
      for (const source of message.sources) {
        await this.addSource({
          messageId,
          conversationId: message.conversationId,
          ...source,
        });
      }
    }

    // Update conversation updatedAt
    await this.updateConversation(message.conversationId, {});

    return messageId;
  }

  async getMessagesByConversation(conversationId: number): Promise<ChatMessage[]> {
    const messages = await this.getAllFromIndex("messages", "conversationId", conversationId);
    
    // Load sources for each message
    for (const message of messages) {
      message.sources = await this.getSourcesByMessage(message.id!);
    }
    
    return messages.sort((a, b) => a.createdAt - b.createdAt);
  }

  async deleteMessagesByConversation(conversationId: number): Promise<void> {
    const messages = await this.getMessagesByConversation(conversationId);
    for (const message of messages) {
      if (message.id) {
        await this.delete("messages", message.id);
      }
    }
  }

  // Sources
  async addSource(source: Omit<SourceDocument, "id" | "createdAt">): Promise<number> {
    const src: SourceDocument = {
      ...source,
      createdAt: Date.now(),
    };
    return this.add("sources", src);
  }

  async getSourcesByMessage(messageId: number): Promise<SourceDocument[]> {
    return this.getAllFromIndex("sources", "messageId", messageId);
  }

  async getSourcesByConversation(conversationId: number): Promise<SourceDocument[]> {
    return this.getAllFromIndex("sources", "conversationId", conversationId);
  }

  async deleteSourcesByConversation(conversationId: number): Promise<void> {
    const sources = await this.getSourcesByConversation(conversationId);
    for (const source of sources) {
      if (source.id) {
        await this.delete("sources", source.id);
      }
    }
  }

  // Generic helpers
  private async add(storeName: string, data: any): Promise<number> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.add(data);
      request.onsuccess = () => resolve(request.result as number);
      request.onerror = () => reject(request.error);
    });
  }

  private async get(storeName: string, key: number): Promise<any> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }

  private async getAll(storeName: string): Promise<any[]> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private async getAllFromIndex(storeName: string, indexName: string, key: number): Promise<any[]> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const index = store.index(indexName);
      const request = index.getAll(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private async put(storeName: string, data: any): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.put(data);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  private async delete(storeName: string, key: number): Promise<void> {
    if (!this.db) throw new Error("Database not initialized");
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}

export const db = new ChatDatabase();
