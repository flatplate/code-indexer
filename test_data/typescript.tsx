// Embedding test file
export interface UserProfile {
	id: number;
	username: string;
	email: string;
	preferences: Record<string, unknown>;
}

export enum UserRole {
	ADMIN = 'ADMIN',
	MODERATOR = 'MODERATOR',
	USER = 'USER',
	GUEST = 'GUEST'
}

export type ValidationResult<T> = {
	isValid: boolean;
	errors?: string[];
	data?: T;
};

export class DatabaseConnection {
	private static instance: DatabaseConnection;
	private connectionString: string;

	private constructor(connectionString: string) {
		this.connectionString = connectionString;
	}

	public static getInstance(connectionString: string): DatabaseConnection {
		if (!DatabaseConnection.instance) {
			DatabaseConnection.instance = new DatabaseConnection(connectionString);
		}
		return DatabaseConnection.instance;
	}

	public async connect(): Promise<void> {
		// Simulated connection logic
	}
}

export const CONFIG = {
	API_VERSION: 'v1',
	MAX_RETRIES: 3,
	TIMEOUT_MS: 5000,
	ENDPOINTS: {
		AUTH: '/auth',
		USERS: '/users',
		POSTS: '/posts'
	}
} as const;

export function debounce<T extends (...args: any[]) => any>(
	func: T,
	wait: number
): (...args: Parameters<T>) => void {
	let timeout: NodeJS.Timeout;
	return (...args: Parameters<T>) => {
		clearTimeout(timeout);
		timeout = setTimeout(() => func(...args), wait);
	};
}

export async function fetchUserData<T extends UserProfile>(
	userId: number
): Promise<ValidationResult<T>> {
	try {
		const response = await fetch(`/api/users/${userId}`);
		const data = await response.json();
		return {
			isValid: true,
			data
		};
	} catch (error) {
		return {
			isValid: false,
			errors: [(error as Error).message]
		};
	}
}

export class EventEmitter<T extends string> {
	private listeners: Map<T, Function[]> = new Map();

	public on(event: T, callback: Function): void {
		const listeners = this.listeners.get(event) || [];
		listeners.push(callback);
		this.listeners.set(event, listeners);
	}

	public emit(event: T, data?: any): void {
		const listeners = this.listeners.get(event) || [];
		listeners.forEach(callback => callback(data));
	}
}

export abstract class BaseRepository<T> {
	protected items: T[] = [];

	abstract create(item: T): Promise<T>;
	abstract update(id: number, item: Partial<T>): Promise<T>;
	abstract delete(id: number): Promise<boolean>;
	abstract findById(id: number): Promise<T | null>;
}

export interface TreeNode<T> {
	value: T;
	left?: TreeNode<T>;
	right?: TreeNode<T>;
}

export function traverse<T>(node: TreeNode<T> | undefined, callback: (value: T) => void): void {
	if (!node) return;
	callback(node.value);
	traverse(node.left, callback);
	traverse(node.right, callback);
}

export class Queue<T> {
	private items: T[] = [];

	enqueue(item: T): void {
		this.items.push(item);
	}

	dequeue(): T | undefined {
		return this.items.shift();
	}

	peek(): T | undefined {
		return this.items[0];
	}
}

export const memoize = <T extends (...args: any[]) => any>(
	fn: T
): ((...args: Parameters<T>) => ReturnType<T>) => {
	const cache = new Map();
	return (...args: Parameters<T>) => {
		const key = JSON.stringify(args);
		if (cache.has(key)) return cache.get(key);
		const result = fn(...args);
		cache.set(key, result);
		return result;
	};
};

export type DeepPartial<T> = {
	[P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export class Observable<T> {
	private value: T;
	private subscribers: ((value: T) => void)[] = [];

	constructor(initialValue: T) {
		this.value = initialValue;
	}

	subscribe(callback: (value: T) => void): () => void {
		this.subscribers.push(callback);
		return () => {
			this.subscribers = this.subscribers.filter(cb => cb !== callback);
		};
	}

	next(value: T): void {
		this.value = value;
		this.subscribers.forEach(callback => callback(value));
	}
}

import React, { useState, useEffect } from 'react';

interface UserCardProps {
	user: UserProfile;
	role: UserRole;
}

const UserCard: React.FC<UserCardProps> = ({ user, role }) => {
	return (
		<div className="user-card">
			<h3>{user.username}</h3>
			<p>{user.email}</p>
			<span className="role-badge">{role}</span>
		</div>
	);
};

const UserList: React.FC = () => {
	const [users, setUsers] = useState<UserProfile[]>([]);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		const loadUsers = async () => {
			try {
				const response = await fetch(`${CONFIG.ENDPOINTS.USERS}`);
				const data = await response.json();
				setUsers(data);
			} catch (error) {
				console.error('Failed to load users:', error);
			} finally {
				setLoading(false);
			}
		};

		loadUsers();
	}, []);

	if (loading) {
		return <div>Loading...</div>;
	}

	return (
		<div className="user-list">
			{users.map(user => (
				<UserCard
					key={user.id}
					user={user}
					role={UserRole.USER}
				/>
			))}
		</div>
	);
};

const SearchInput: React.FC<{ onSearch: (term: string) => void }> = ({ onSearch }) => {
	const debouncedSearch = debounce((term: string) => {
		onSearch(term);
	}, 300);

	return (
		<input
			type="text"
			placeholder="Search users..."
			onChange={(e) => debouncedSearch(e.target.value)}
		/>
	);
};

const UserDashboard: React.FC = () => {
	const [searchTerm, setSearchTerm] = useState('');

	return (
		<div className="dashboard">
			<SearchInput onSearch={setSearchTerm} />
			<UserList />
		</div>
	);
};

export const init = () => {
	const connection = DatabaseConnection.getInstance('localhost:5432');
	connection.connect();
}

export { UserCard, UserList, SearchInput, UserDashboard };
