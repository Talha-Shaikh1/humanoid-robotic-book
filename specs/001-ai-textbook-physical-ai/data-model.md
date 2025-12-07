# Data Model: AI-Native Textbook on Physical AI & Humanoid Robotics

## Core Entities

### User
**Description**: Represents students, educators, and professionals using the textbook
**Fields**:
- id: UUID (primary key)
- email: String (unique, required)
- name: String (required)
- role: Enum (student, educator, professional)
- created_at: DateTime (required)
- updated_at: DateTime (required)
- preferences: JSON (user preferences for personalization)
- auth_provider: String (auth provider if using OAuth)

### ContentModule
**Description**: Represents the core content modules (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA)
**Fields**:
- id: UUID (primary key)
- title: String (required)
- description: Text
- module_type: Enum (ros2, gazebo_unity, nvidia_isaac, vla)
- week_number: Integer (1-13, for weekly breakdown)
- content: Text (MDX content)
- prerequisites: JSON (list of prerequisite module IDs)
- learning_outcomes: JSON (list of learning outcomes)
- created_at: DateTime (required)
- updated_at: DateTime (required)

### InteractiveElement
**Description**: Represents interactive elements like code examples, URDF snippets, quizzes
**Fields**:
- id: UUID (primary key)
- type: Enum (code_example, urdf_snippet, quiz, exercise)
- title: String (required)
- content: Text (the actual code/quiz content)
- module_id: UUID (foreign key to ContentModule)
- difficulty: Enum (beginner, intermediate, advanced)
- language: String (for code examples)
- created_at: DateTime (required)
- updated_at: DateTime (required)

### Assessment
**Description**: Represents quizzes and exercises for user evaluation
**Fields**:
- id: UUID (primary key)
- title: String (required)
- description: Text
- module_id: UUID (foreign key to ContentModule)
- questions: JSON (list of questions with answers)
- max_score: Integer (required)
- time_limit: Integer (in minutes, null if untimed)
- created_at: DateTime (required)
- updated_at: DateTime (required)

### UserProgress
**Description**: Tracks user progress through the textbook content
**Fields**:
- id: UUID (primary key)
- user_id: UUID (foreign key to User)
- module_id: UUID (foreign key to ContentModule)
- assessment_id: UUID (foreign key to Assessment, nullable)
- progress: Float (0.0 to 1.0, percentage complete)
- score: Float (for assessments, 0.0 to 1.0)
- completed_at: DateTime (nullable)
- created_at: DateTime (required)
- updated_at: DateTime (required)

### ChatSession
**Description**: Stores chatbot conversation sessions for RAG functionality
**Fields**:
- id: UUID (primary key)
- user_id: UUID (foreign key to User, nullable for anonymous)
- session_title: String (auto-generated from first query)
- created_at: DateTime (required)
- updated_at: DateTime (required)

### ChatMessage
**Description**: Individual messages in a chatbot conversation
**Fields**:
- id: UUID (primary key)
- session_id: UUID (foreign key to ChatSession)
- role: Enum (user, assistant)
- content: Text (required)
- source_documents: JSON (list of source document IDs used by RAG)
- created_at: DateTime (required)

### TranslationCache
**Description**: Caches translated content to improve performance and reduce API costs
**Fields**:
- id: UUID (primary key)
- original_content_id: UUID (ID of original content)
- original_content_type: String (type of content being translated)
- original_text: Text (required)
- translated_text: Text (required)
- target_language: String (required, e.g., 'ur' for Urdu)
- created_at: DateTime (required)

## Relationships

- User 1 --- * UserProgress (user has many progress records)
- ContentModule 1 --- * UserProgress (module has many progress records)
- ContentModule 1 --- * InteractiveElement (module has many interactive elements)
- ContentModule 1 --- * Assessment (module has many assessments)
- User 1 --- * ChatSession (user has many chat sessions)
- ChatSession 1 --- * ChatMessage (session has many messages)
- User 1 --- 1 UserProfile (user has one profile for personalization)

## Validation Rules

### User
- Email must be valid email format
- Role must be one of the defined enum values
- Name must not be empty

### ContentModule
- Title must not be empty
- Week number must be between 1 and 13
- Module type must be one of the defined enum values

### InteractiveElement
- Type must be one of the defined enum values
- Module ID must reference an existing ContentModule

### Assessment
- Max score must be positive
- Questions must contain valid question/answer format
- Module ID must reference an existing ContentModule

### UserProgress
- Progress must be between 0.0 and 1.0
- Score must be between 0.0 and 1.0 (if present)
- User ID must reference an existing User
- Module ID must reference an existing ContentModule

## State Transitions

### UserProgress
- `not_started` → `in_progress` → `completed`
- Progress percentage increases as user interacts with content
- Score is set when assessment is completed

### ChatSession
- Created when user starts a new conversation
- Updated as new messages are added
- Session is considered active until a certain time period passes without new messages